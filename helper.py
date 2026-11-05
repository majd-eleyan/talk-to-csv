# حتوي على دوال مساعدة تستخدمها باقي أجزاء المشروع
# دوال مساعدة: قراءة JSON من رد الـ LLM، بناء الـ schema، تجهيز النتيجة، اختيار الشارت

import ast
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

import config


# ---------------------------------------------------------------
#استخراج جيسون من رد الـ مودل 
# ---------------------------------------------------------------
def extract_json_object(text):
    text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    # نمشي حرف حرف لحد ما نلاقي نهاية الـ المطلوب (مع تجاهل الأقواس داخل النصوص)
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(text[start:i + 1])
                if not isinstance(data, dict):
                    raise ValueError("LLM response is not a JSON object")
                return data

    raise ValueError("Unbalanced JSON object in LLM response")

# وظيفتها تنظيف كود بايثون الذي يرجعه الـ LLM.
def clean_code(code):
    if not isinstance(code, str):
        return ""
    code = re.sub(r"^```(?:python)?", "", code.strip())
    return re.sub(r"```$", "", code).strip()


# ---------------------------------------------------------------
# 2) الـ Schema 
# ---------------------------------------------------------------
def build_schema(df):
    return {
        "rows": len(df),
        "columns": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isna().sum()),
            }
            for col in df.columns
        ],
        "sample_rows": df.head(config.SAMPLE_ROWS_FOR_LLM).to_dict("records"),
    }


# ---------------------------------------------------------------
#  كشف الأعمدة غير الموجودة (بدون تنفيذ الكود)
# الاعمدة الي بتستخدم في الكود ومو موجودة في الداتا
# ---------------------------------------------------------------
def find_missing_columns(code, df):
    """نبحث عن df['xxx'] وxxx مو موجود بالداتا (ومو عمود جديد الكود أنشأه)"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    referenced, created = [], set()
    for node in ast.walk(tree):
        # df['col'] = ...  (عمود جديد ينشئه الكود)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == "df" and isinstance(t.slice, ast.Constant)):
                    created.add(t.slice.value)
        # df.assign(new_col=...)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "assign"):
            created.update(kw.arg for kw in node.keywords if kw.arg)
        # df['col'] أو df[['a','b']] (قراءة)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "df" and isinstance(node.ctx, ast.Load)):
            s = node.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                referenced.append(s.value)
            elif isinstance(s, ast.List):
                referenced += [e.value for e in s.elts
                               if isinstance(e, ast.Constant) and isinstance(e.value, str)]

    missing = [c for c in referenced if c not in df.columns and c not in created]
    return list(dict.fromkeys(missing))


"""أعمدة استخدمها الكود وفيها قيم مفقودة (نعرض ملاحظة للمستخدم)"""
def columns_used_with_missing(code, report):
    notes = []
    for col, (count, _) in report.missing.items():
        if count > 0 and (f"'{col}'" in code or f'"{col}"' in code):
            notes.append(f"{col} has {count} missing value(s); Pandas ignored them in the calculation.")
    return notes


# ---------------------------------------------------------------
#  تجهيز النتيجة للعرض
# ---------------------------------------------------------------
def normalize_result(raw):
    """بترجع (النوع, القيمة). النوع: scalar / series / dataframe"""
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, pd.DataFrame):
        return "dataframe", raw
    if isinstance(raw, pd.Series):
        return "series", raw
    if isinstance(raw, (int, float, str, bool, pd.Timestamp)):
        return "scalar", raw
    if isinstance(raw, (list, tuple)):
        return "series", pd.Series(list(raw))
    if isinstance(raw, dict):
        return "series", pd.Series(raw)
    raise ValueError(
        f"`result` must be a number, text, Series or DataFrame (got {type(raw).__name__})."
    )

# دالة بسيطة لتنسيق الأرقام
def format_scalar(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


"""ملخص قصير للـ history (بدون تخزين الداتا نفسها)"""
def summarize_result(kind, value):
    if kind == "scalar":
        return format_scalar(value)
    if kind == "series":
        if len(value) == 0:
            return "Empty result"
        first = value.iloc[0]
        first_txt = format_scalar(first.item() if isinstance(first, np.generic) else first)
        return f"{len(value)} rows; first: {value.index[0]} = {first_txt}"
    return f"Table with {len(value)} rows x {value.shape[1]} columns"


# ---------------------------------------------------------------
# اختيار الشارت المناسب حسب شكل النتيجة (Plotly)
# ---------------------------------------------------------------
def _to_frame(result):
    if isinstance(result, pd.DataFrame):
        frame = result.copy()
        if not isinstance(frame.index, pd.RangeIndex) or frame.index.name:
            frame = frame.reset_index()
    else:
        name = str(result.name) if result.name is not None else "value"
        if name in [str(n) for n in result.index.names]:
            name += "_value"
        frame = result.rename(name).reset_index()
        if "index" in frame.columns:
            frame = frame.rename(columns={"index": "category"})
    frame.columns = [str(c) for c in frame.columns]
    for c in frame.columns:  # فترات زمنية (Period) -> تواريخ
        if isinstance(frame[c].dtype, pd.PeriodDtype):
            frame[c] = frame[c].dt.to_timestamp()
    return frame

# بناء رسمة مناسب بناءً على شكل النتيجة والسؤال.
def choose_chart(result, question=""):
    """
    - مقارنة بين فئات        -> Bar
    - بيانات عبر الزمن        -> Line
    - توزيع (distribution)   -> Histogram
    - نسب (share/percent)    -> Pie (بس إذا الفئات قليلة)
    - علاقة بين رقمين        -> Scatter
    بترجع figure أو None إذا ما في شارت مناسب
    """
    q = question.lower()
    try:
        # Histogram للتوزيع: النتيجة قيم رقمية خام
        if (isinstance(result, pd.Series) and is_numeric_dtype(result) and len(result) > 1
                and any(w in q for w in ("distribution", "histogram"))):
            fig = px.histogram(x=result.dropna(), title="Distribution")
            fig.update_layout(xaxis_title=str(result.name or "value"), yaxis_title="Count")
            return fig

        if not isinstance(result, (pd.Series, pd.DataFrame)) or len(result) == 0:
            return None

        frame = _to_frame(result)
        cols = list(frame.columns)
        numeric = [c for c in cols if is_numeric_dtype(frame[c])]
        dates = [c for c in cols if is_datetime64_any_dtype(frame[c])]
        others = [c for c in cols if c not in numeric and c not in dates]
        if not numeric and not dates:
            return None

        # Line: بيانات عبر الزمن
        if dates:
            y_cols = [c for c in numeric if c != dates[0]]
            if not y_cols:
                return None
            color = others[0] if others and frame[others[0]].nunique() <= 10 else None
            fig = px.line(frame, x=dates[0], y=y_cols[0], color=color, markers=True)
            fig.update_layout(xaxis_title=dates[0], yaxis_title=y_cols[0])
            return fig

        # Scatter: علاقة بين عمودين رقميين
        if not others and len(numeric) >= 2 and any(
                w in q for w in ("relationship", "correlation", "scatter", " vs ", "versus")):
            fig = px.scatter(frame, x=numeric[0], y=numeric[1])
            fig.update_layout(xaxis_title=numeric[0], yaxis_title=numeric[1])
            return fig

        if len(frame) > 50:
            return None  # كثير على شارت واضح

        x_col = others[0] if others else cols[0]
        y_options = [c for c in numeric if c != x_col]
        if not y_options:
            return None
        y_col = y_options[0]

        # Pie: بس لما الفئات قليلة والقيم موجبة والسؤال عن نسب
        if (any(w in q for w in ("share", "proportion", "percent", "pie")) and len(others) == 1
                and 2 <= len(frame) <= 8 and (frame[y_col] >= 0).all()):
            return px.pie(frame, names=x_col, values=y_col)

        # Bar: مقارنة (الافتراضي)
        color = others[1] if len(others) >= 2 else None
        fig = px.bar(frame, x=x_col, y=y_col, color=color, barmode="group")
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        return fig
    except Exception:
        return None  # الشارت إضافي: إذا فشل ما نوقف التطبيق
