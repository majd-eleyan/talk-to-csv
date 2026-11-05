# بيحوّل النتيجة الحقيقية (من Pandas) لجواب نصي طبيعي مثل الشات
# القاعدة: الأرقام ما بتجي من الـ LLM. المودل بس بيصيغ الجملة، وإحنا بنتأكد إن كل رقم بالجملة موجود بالنتيجة

import re

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

import config
from helper import format_scalar
from llm import LLMError, ask_llm
from prompts import build_answer_prompt

MAX_ROWS_FOR_LLM = 10
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# قيمة وحدة بنفس تنسيق الجداول اللي بتظهر للمستخدم
def _fmt(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        value = value.item()
    return format_scalar(value)


# النتيجة كنص بسيط نبعته للمودل (أول 10 صفوف بس)
def result_to_text(kind, value):
    if kind == "scalar":
        return _fmt(value), 1
    if kind == "series":
        rows = [f"{idx}: {_fmt(v)}" for idx, v in list(value.items())[:MAX_ROWS_FOR_LLM]]
        return "\n".join(rows), len(value)
    rows = []
    for _, row in value.head(MAX_ROWS_FOR_LLM).iterrows():
        rows.append(", ".join(f"{col}: {_fmt(row[col])}" for col in value.columns))
    return "\n".join(rows), len(value)


# جواب بسيط بدون LLM (للاحتياط لو المودل فشل أو غلط بالأرقام)
def fallback_answer(kind, value):
    if kind == "scalar":
        return f"The answer is {format_scalar(value)}."
    text, count = result_to_text(kind, value)
    if count == 0:
        return "No matching results were found."
    if count == 1 or (kind == "series" and count <= 3):
        return text.replace("\n", "; ") + "."
    return f"I found {count} results. Here they are."


def _numbers(text):
    text = text.translate(_ARABIC_DIGITS)
    found = set()
    for m in _NUMBER.findall(text):
        m = m.replace(",", "")
        if "." in m:
            m = m.rstrip("0").rstrip(".")
        found.add(m)
    return found


# كل رقم بالجملة لازم يكون من النتيجة أو من السؤال أو عدد الصفوف
def numbers_are_safe(answer, allowed_text, count):
    return _numbers(answer) <= (_numbers(allowed_text) | {str(count)})


def write_answer(question, clarifications, kind, value):
    empty = (kind != "scalar" and len(value) == 0)
    if empty or not config.NATURAL_LANGUAGE_ANSWERS:
        return fallback_answer(kind, value)

    result_text, count = result_to_text(kind, value)
    allowed = result_text + " " + question + " " + " ".join(
        f"{c['question']} {c['answer']}" for c in clarifications)
    system_prompt, user_prompt = build_answer_prompt(question, clarifications, result_text, count)
    try:
        answer = ask_llm(system_prompt, user_prompt).strip()
    except LLMError:
        return fallback_answer(kind, value)   # فشل الاتصال؟ الجواب الأساسي ما بيضيع

    if not answer or not numbers_are_safe(answer, allowed, count):
        return fallback_answer(kind, value)
    return answer
