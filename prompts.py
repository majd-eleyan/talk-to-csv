
# LLM generates code
#         ↓
# Validator
#         ↓
# Execution
#         ↓
# ERROR
#         ↓
# LLM correction
#         ↓
# Validator AGAIN
#         ↓
# Execution

import json

import config
from code_validator import ALLOWED_ATTRIBUTES, ALLOWED_METHODS
from helper import build_schema

#  حدود قدرات النظام
CAPABILITIES = """
Supported analysis: counting, filtering, average, sum, min/max, grouped comparison,
top-N, sorting, basic aggregations, and basic time-based analysis when a date column exists.
NOT supported: predictions/forecasting, machine learning, external/web information,
or anything that needs data that is not in the dataset.
"""


def _base_system_prompt(df, quality_summary, can_ask=True):
    schema = build_schema(df)

    # لو وصلنا لأقصى عدد أسئلة توضيح، بنشيل هاد الخيار من البرومت
    if can_ask:
        clarification_action = """
   Ambiguous or incomplete question (see the rules for clarification below):
   {"action": "ask_clarification", "clarification_question": "ONE short question", "options": ["option A", "option B", "option C"]}
"""
        clarification_rules = """
Rules for clarification:
- Ask ONLY when the question has more than one reasonable meaning that would give different answers
  (e.g. "best product": by revenue, by quantity or by number of orders?), or when an important detail is missing
  (which column, which time period, how many items).
- Do NOT ask when the question is already clear, or when the schema leaves only one sensible meaning.
- Ask ONE short question in plain business language. Give 2 to 4 short "options" the user can pick from.
"""
    else:
        clarification_action = ""
        clarification_rules = """
You already asked enough questions. Do NOT use "ask_clarification" now.
Pick the most reasonable meaning and say which one you chose in "explanation".
"""

    return f"""
You are a Data Analyst Agent. You work with a pandas DataFrame named `df`
(already cleaned) and the `pd` module.

DATASET SCHEMA (real data, do not invent columns):
{json.dumps(schema, indent=2, default=str)}

DATA QUALITY SUMMARY (mention it only if it affects the answer):
{quality_summary}

{CAPABILITIES}

Your job:
1. Understand the user's question.
2. Decide ONE action and reply with ONLY one valid JSON object (no markdown, no prose):

   Clear question that the data can answer:
   {{"action": "generate_code", "code": "PANDAS_CODE", "explanation": "one plain-English sentence saying what the code does"}}
{clarification_action}
   Question outside the scope (prediction, ML, external info, missing data):
   {{"action": "unsupported", "reason": "short explanation"}}

   Question about a column that does NOT exist in the schema:
   {{"action": "unsupported", "reason": "column not found", "missing_column": "the_name_the_user_asked_for"}}
{clarification_rules}
Rules for "code":
- Use ONLY the exact column names from the schema. Never invent columns.
- Use only `df` and `pd`. No imports, no loops, no lambda, no print(), no file access.
- Store the final answer in a variable named `result`.
- `result` must be a number, text, Series or DataFrame.
- Use single quotes inside the code so the JSON stays valid.
- Missing values: let pandas ignore them. Do NOT fill or invent values unless asked.
- Never write numbers you calculated yourself. Pandas will calculate the real result.
- You may compute derived values with vectorized operations, e.g. df['Price'] * df['Quantity'].
- Allowed methods: {", ".join(sorted(ALLOWED_METHODS))}
- Allowed attributes: {", ".join(sorted(ALLOWED_ATTRIBUTES))}
- Examples:
  result = df['Price'].mean()
  result = df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
  result = df.groupby('Product')['Revenue'].sum().nlargest(5)

Return ONLY the JSON object.
"""


# بنكتب السؤال الأصلي + أسئلة التوضيح وإجابات المستخدم عليها
def _question_block(question, clarifications):
    text = f"User question: {question}"
    if clarifications:
        text += "\n\nYou already asked the user for clarification:"
        for item in clarifications:
            text += f"\n- You asked: {item['question']}\n  User answered: {item['answer']}"
        text += (
            "\n\nUse these answers to understand the original question and answer it now. "
            "Do not ask again about something the user already answered. "
            "If the last answer is clearly a NEW question and not an answer, answer the new question instead."
        )
    return text


# system_prompt + user_prompt
def build_decision_prompt(question, df, quality_summary, clarifications=None):
    clarifications = clarifications or []
    can_ask = len(clarifications) < config.MAX_CLARIFICATIONS
    system_prompt = _base_system_prompt(df, quality_summary, can_ask)
    user_prompt = _question_block(question, clarifications)
    return system_prompt, user_prompt


def build_correction_prompt(question, df, quality_summary, failed_code, error_message, clarifications=None):
    """بنرجع للـ LLM الخطأ عشان يصحح الكود (بيتم فحصه من جديد قبل التنفيذ)"""
    clarifications = clarifications or []
    # وقت التصحيح ما بنسأل المستخدم أسئلة جديدة
    system_prompt = _base_system_prompt(df, quality_summary, can_ask=False)
    user_prompt = f"""{_question_block(question, clarifications)}

Your previous code failed.

Previous code:
{failed_code}

Error:
{error_message}

Write corrected code. Use only allowed methods and existing columns.
If the error shows that the needed column does not exist, answer with the "unsupported" action and "missing_column".
Reply with ONLY the JSON object in the same format as before.
"""
    return system_prompt, user_prompt


# برومت صياغة الجواب النهائي بجملة طبيعية (الأرقام من النتيجة فقط)
def build_answer_prompt(question, clarifications, result_text, row_count):
    system_prompt = """You write the final answer for a data analysis chat.
Write 1 to 3 short, friendly sentences that answer the user's question using ONLY the RESULT.

Rules:
- Use the numbers and names exactly as written in RESULT. Never invent, change, round or calculate new numbers.
- Do NOT mention code, Pandas, DataFrame, columns or how the result was calculated.
- Answer in the same language as the user's question.
- Do not use tables, bullet lists or markdown headings.
- If RESULT has several rows, mention the most important one or two and say the full list is shown below.
- Reply with the answer text only."""
    user_prompt = f"User question: {question}"
    for item in clarifications or []:
        user_prompt += f"\nClarification asked: {item['question']}\nUser answered: {item['answer']}"
    shown = min(row_count, 10)
    note = f" (showing {shown} of {row_count} rows)" if row_count > shown else ""
    user_prompt += f"\n\nRESULT{note}:\n{result_text}"
    return system_prompt, user_prompt
