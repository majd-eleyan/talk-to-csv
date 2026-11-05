
#  الملف الرئيسي:
# Question -> LLM -> Decision -> Validate -> Execute -> (Correct) -> Result

import config
from code_exec import ExecutionError, UnsafeCodeError, run_pandas_code
from helper import (
    choose_chart,
    clean_code,
    columns_used_with_missing,
    extract_json_object,
    find_missing_columns,
    normalize_result,
    summarize_result,)
from answer_writer import write_answer
from llm import LLMError, ask_llm
from prompts import build_correction_prompt, build_decision_prompt


# نسأل الـ الذكاء ونحوّل الرد إلى جيسون
def _get_decision(system_prompt, user_prompt):
    raw = ask_llm(system_prompt, user_prompt)
    try:
        return extract_json_object(raw)
    except ValueError:
        raise
    except Exception as e:  # json.loads فشل
        raise ValueError(f"Invalid JSON from the model: {e}") from e


# خيارات الإجابة السريعة اللي بيقترحها المودل: بنتأكد إنها نصوص قصيرة وعددها قليل
def _clean_options(options):
    if not isinstance(options, list):
        return []
    clean = [str(o).strip() for o in options if isinstance(o, (str, int, float)) and str(o).strip()]
    return [o for o in clean if len(o) <= 60][:4]


# شكل موحّد للنتيجة اللي بيرجعها النظام للواجهة
def _reply(question, action, message="", summary="", **extra):
    return {"question": question, "action": action, "message": message,
            "summary": summary or message, **extra}

# الرد لو في عمود مش موجود بالسؤال
def _missing_column_reply(question, missing, df, code=None):
    names = ", ".join(f'"{m}"' for m in missing)
    available = ", ".join(df.columns)
    message = (f"I couldn't find a column named {names}.\n\n"
               f"Available columns: {available}")
    return _reply(question, "missing_column", message,
                  summary=f"Column not found: {', '.join(missing)}",
                  code=code, missing=missing)

# الدالة الرئيسية
def ask_agent(question, df, report, clarifications=None):
    # clarifications: أسئلة التوضيح السابقة وإجابات المستخدم عليها [{"question":..., "answer":...}]
    clarifications = clarifications or []
    quality_summary = report.summary_for_llm()

    # 1) Understand + Decide
    system_prompt, user_prompt = build_decision_prompt(question, df, quality_summary, clarifications)
    try:
        decision = _get_decision(system_prompt, user_prompt)
    except LLMError as e:
        return _reply(question, "error", str(e), summary="AI service error")
    except ValueError:
        return _reply(question, "error",
                      "The AI model returned an answer I could not understand "
                      "(invalid JSON). Please try again or rephrase the question.",
                      summary="Invalid AI answer")

    corrections = 0
    last_code, last_error = None, ""

    while True:
        action = decision.get("action")

        # ---- سؤال غامض: نسأل المستخدم ولا نخمّن ----
        if action == "ask_clarification":
            # وصلنا لأقصى عدد أسئلة؟ ما بنسأل أكثر عشان ما ندخل بحلقة
            if len(clarifications) >= config.MAX_CLARIFICATIONS:
                return _reply(question, "unsupported",
                              "I still can't tell exactly what you need. "
                              "Please write the full question in one message, for example with the column name.")
            text = decision.get("clarification_question") or "Could you clarify your question?"
            return _reply(question, "ask_clarification", text,
                          options=_clean_options(decision.get("options")))

        # ---- سؤال خارج النطاق أو عمود غير موجود ----
        if action == "unsupported":
            missing = decision.get("missing_column")
            if missing:
                missing = [missing] if isinstance(missing, str) else list(missing)
                return _missing_column_reply(question, missing, df)
            reason = decision.get("reason") or "This question is outside the scope of the system."
            return _reply(question, "unsupported", reason)
        
        if action != "generate_code":
            return _reply(question, "error",
                          "The AI model returned an unknown action. Please try again.",
                          summary="Unknown AI action")

        # ---- Generate Code ----
        code = clean_code(decision.get("code"))
        explanation = decision.get("explanation") or ""
        last_code = code

        # عمود غير موجود؟ نكتشفه قبل التنفيذ ونرجع رسالة واضحة
        missing = find_missing_columns(code, df)
        if missing:
            return _missing_column_reply(question, missing, df, code=code)

        # ---- Validate + Execute ----
        try:
            raw = run_pandas_code(df, code)  # جوّاها AST validation
            kind, value = normalize_result(raw)
            return _reply(
                question, "generate_code",
                summary=summarize_result(kind, value),
                code=code, explanation=explanation,
                answer=write_answer(question, clarifications, kind, value),
                result_kind=kind, result=value,
                chart=choose_chart(value, question),
                notes=columns_used_with_missing(code, report),
                corrections=corrections,
                clarifications=clarifications,
            )
        except UnsafeCodeError as e:
            last_error = "Rejected by the security validator: " + "; ".join(e.errors)
        except ExecutionError as e:
            last_error = str(e)
            # على عمود مو موجود -> رسالة واضحة بدل خطأ تقنيKeyError 
            if isinstance(e.original, KeyError) and e.original.args:
                key = e.original.args[0]
                if isinstance(key, str) and key not in df.columns:
                    return _missing_column_reply(question, [key], df, code=code)
        except ValueError as e:  # نتيجة بنوع غير مدعوم
            last_error = str(e)

        # ---- Error Recovery: أقصى محاولتين تصحيح ----
        if corrections >= config.MAX_CODE_RETRIES:
            return _reply(
                question, "error",
                f"I could not produce working code after {config.MAX_CODE_RETRIES} correction "
                f"attempts. Last error: {last_error}\n\nTry rephrasing your question.",
                summary="Analysis failed", code=last_code, corrections=corrections)

        corrections += 1
        system_prompt, user_prompt = build_correction_prompt(
            question, df, quality_summary, last_code, last_error, clarifications)
        try:
            decision = _get_decision(system_prompt, user_prompt)
        except (LLMError, ValueError) as e:
            return _reply(question, "error",
                          f"The correction attempt failed: {e}",
                          summary="Correction failed", code=last_code, corrections=corrections)
