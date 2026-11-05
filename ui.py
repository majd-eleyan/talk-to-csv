# ui.py
# كل اللي إله علاقة بالشكل: الستايل، الأزرار، عرض ردود الـ agent داخل الشات
# (منطق التحليل كله بقي في full_system.py وما لمسناه)

import html

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from data_quality import READY, READY_WITH_WARNINGS
from helper import format_scalar

# ألوان الرسومات (نفس ألوان الـ theme)
PALETTE = ["#2F4BDB", "#E8A317", "#0E9F8E", "#C2410C", "#6D5BD0", "#64748B"]

CSS = """
<style>
/* عرض مريح للشات */
[data-testid="stMainBlockContainer"] { max-width: 960px; padding-top: 2.2rem; }

/* عنوان الصفحة */
.page-title { font-size: 1.55rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; }
.page-sub { color: #667085; font-size: 0.95rem; margin: 0.15rem 0 1.2rem 0; }

/* رسالة المستخدم: فقاعة على اليمين */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse; background: #E8EDFF; border: 1px solid #D5DDFB; border-radius: 14px;
  width: fit-content; max-width: 82%; margin-left: auto; padding: 0.6rem 1rem;
}
/* رد الـ agent: بطاقة بيضاء (الأفاتار المخصص = رسالة الـ agent) */
[data-testid="stChatMessage"] {
  background: #FFFFFF; border: 1px solid #E1E5EC; border-radius: 14px; padding: 0.9rem 1.1rem;
}

/* الرقم الكبير للنتيجة */
.answer-value { font-size: 2.1rem; font-weight: 600; letter-spacing: -0.02em; color: #101828; line-height: 1.15; }
.answer-text  { font-size: 1.1rem; font-weight: 500; color: #101828; }
.muted { color: #667085; font-size: 0.85rem; }

/* حالة الداتا */
.pill { display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.78rem; font-weight: 500; }
.pill.ok   { background: #DCFCE7; color: #166534; }
.pill.warn { background: #FEF3C7; color: #92400E; }
.pill.bad  { background: #FEE2E2; color: #991B1B; }

/* شبكة الأرقام الصغيرة بالـ sidebar */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin: 0.8rem 0 1rem 0; }
.stat { background: #182A4D; border-radius: 10px; padding: 0.55rem 0.75rem; }
.stat .v { font-size: 1.2rem; font-weight: 600; color: #FFFFFF; }
.stat .k { font-size: 0.74rem; color: #9FB0D3; }
.brand { font-size: 1.15rem; font-weight: 600; color: #FFFFFF; margin: 0.2rem 0 0 0; }
.brand-sub { font-size: 0.82rem; line-height: 1.35; color: #9FB0D3; margin: 0.15rem 0 1.1rem 0; }
.file-name { font-size: 0.9rem; font-weight: 500; color: #FFFFFF; word-break: break-all; margin-bottom: 0.4rem; }

/* الشاشة الأولى (بدون ملف) */
.hero-title { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; margin: 2.5rem 0 0.4rem 0; }
.hero-text  { color: #475467; font-size: 1.02rem; max-width: 40rem; line-height: 1.55; }

/* إطار كحلي حول مربع الكتابة (نفس لون الـ sidebar) */
[data-testid="stChatInput"] > div {
  border: 2px solid #0F1B33; border-radius: 14px;
}
[data-testid="stChatInput"] > div:focus-within { border-color: #2F4BDB; }
</style>
"""


# بنطبّق الستايل مرة وحدة بأول الصفحة
def apply_style():
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = PALETTE
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(subtitle):
    st.markdown(
        f'<div class="page-title">Talk to CSV</div><div class="page-sub">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# الـ sidebar
# ---------------------------------------------------------------
def sidebar_brand():
    st.markdown(
        '<div class="brand">Talk to CSV</div>'
        '<div class="brand-sub">Ask your CSV questions in plain English</div>',
        unsafe_allow_html=True,
    )


def status_pill(status):
    if status == READY:
        label, kind = "Ready", "ok"
    elif status == READY_WITH_WARNINGS:
        label, kind = "Ready with warnings", "warn"
    else:
        label, kind = "Rejected", "bad"
    return f'<span class="pill {kind}">{label}</span>'


def sidebar_dataset(file_name, report):
    st.markdown(
        f'<div class="file-name">{html.escape(file_name)}</div>{status_pill(report.status)}',
        unsafe_allow_html=True,
    )
    if report.cleaned_df is None:
        return
    st.markdown(
        '<div class="stat-grid">'
        f'<div class="stat"><div class="v">{report.cleaned_rows:,}</div><div class="k">Rows</div></div>'
        f'<div class="stat"><div class="v">{report.cleaned_cols}</div><div class="k">Columns</div></div>'
        f'<div class="stat"><div class="v">{report.overall_missing_rate:.1%}</div><div class="k">Missing values</div></div>'
        f'<div class="stat"><div class="v">{report.duplicate_rows}</div><div class="k">Duplicates removed</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


# نافذة تفاصيل الداتا (بدل ما نحشر كل شي بصفحة الشات)
@st.dialog("Dataset details", width="large")
def show_dataset_details(report):
    cleaned = report.cleaned_df
    tab_preview, tab_columns, tab_quality = st.tabs(["Preview", "Columns", "Data quality report"])
    with tab_preview:
        st.dataframe(pretty_table(cleaned.head(10)), width="stretch")
    with tab_columns:
        table = pd.DataFrame({
            "Column": cleaned.columns,
            "Type": [str(t) for t in cleaned.dtypes],
            "Missing": [report.missing[c][0] for c in cleaned.columns],
            "Missing %": [f"{report.missing[c][1]:.1%}" for c in cleaned.columns],
        })
        st.dataframe(table, hide_index=True, width="stretch")
    with tab_quality:
        st.markdown(report.to_markdown())


# ---------------------------------------------------------------
# أسئلة مقترحة مبنية على أعمدة الملف (بتشتغل لأي CSV)
# ---------------------------------------------------------------
def suggest_questions(df):
    numeric = [c for c in df.columns if is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    dates = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    cats = [c for c in df.columns
            if c not in numeric and c not in dates and 2 <= df[c].nunique() <= 30]

    # نفضّل عمود قيمته مالية (للمجموع) وعمود فئات واضح (للمقارنة) لو موجودين
    def pick(columns, words):
        for w in words:
            for c in columns:
                if w in c.lower():
                    return c
        return columns[0] if columns else None

    num = pick(numeric, ("revenue", "sales", "amount", "total", "profit", "cost", "price"))
    cat = pick(cats, ("category", "product", "region", "segment", "type", "department", "customer", "country", "city"))

    questions = ["How many rows are there?"]
    if num:
        questions.append(f"What is the average {num}?")
    if num and cat:
        questions.append(f"Which {cat} has the highest total {num}?")
        questions.append(f"What are the top 5 {cat} values by total {num}?")
    if num and dates:
        questions.append(f"Show the monthly total {num} over time.")
    if cat:
        questions.append(f"Which {cat} is the best?")  # سؤال غامض: الـ agent رح يسأل توضيح
    return questions[:6]


def starter_buttons(df):
    st.markdown('<div class="muted">Try one of these, or type your own question.</div>', unsafe_allow_html=True)
    questions = suggest_questions(df)
    cols = st.columns(2)
    for i, q in enumerate(questions):
        if cols[i % 2].button(q, key=f"starter_{i}", width="stretch"):
            st.session_state.queued = q
            st.rerun()


# ---------------------------------------------------------------
# تنسيق الجداول: فواصل الآلاف، منزلتين للكسور، تاريخ بدون ساعة
# ---------------------------------------------------------------
def pretty_table(df):
    try:
        df = df.copy()
        for c in df.columns:
            col = df[c]
            if is_datetime64_any_dtype(col) and (col.dropna().dt.normalize() == col.dropna()).all():
                df[c] = col.dt.strftime("%Y-%m-%d")
        formats = {}
        for c in df.columns:
            if pd.api.types.is_bool_dtype(df[c]):
                continue
            if pd.api.types.is_integer_dtype(df[c]):
                formats[c] = "{:,}"
            elif pd.api.types.is_float_dtype(df[c]):
                formats[c] = "{:,.2f}"
        return df.style.format(formats, na_rep="-") if formats else df
    except Exception:
        return df  # لو التنسيق فشل نعرض الجدول كما هو


# شكل الرسم: ارتفاع مناسب وبدون شريط الأدوات
def show_chart(fig, key):
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=32, b=8))
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


# ---------------------------------------------------------------
# عرض رد الـ agent داخل الشات
# ---------------------------------------------------------------
def render_assistant(r, idx, is_last, pending):
    action = r["action"]

    if action == "ask_clarification":
        st.markdown(r["message"])
        options = r.get("options") or []
        # أزرار الإجابة السريعة بس لآخر رسالة وبس لو النظام لسا مستني الجواب
        if is_last and pending and options:
            cols = st.columns(len(options))
            for j, opt in enumerate(options):
                if cols[j].button(opt, key=f"opt_{idx}_{j}", width="stretch"):
                    st.session_state.queued = opt
                    st.rerun()
        if is_last and pending:
            st.caption("Pick an option or type your own answer below.")
        return

    if action in ("unsupported",):
        st.warning(r["message"])
    elif action in ("missing_column", "error"):
        st.error(r["message"])
    else:  # generate_code نجح
        # الجواب نصي مثل الشات (بنهرّب $ عشان ما يتحول لمعادلة)
        st.markdown(r.get("answer", "").replace("$", "\\$"))
        kind, value = r["result_kind"], r["result"]
        # الجدول بنعرضه بس لو النتيجة أكثر من صف (الصف الواحد الجملة بتكفي)
        if kind == "series" and len(value) > 1:
            st.dataframe(pretty_table(value.to_frame(name=str(value.name or "value"))), width="stretch")
        elif kind == "dataframe" and len(value) > 1:
            st.dataframe(pretty_table(value), width="stretch")
        if r.get("chart") is not None:
            show_chart(r["chart"], key=f"chart_{idx}")
        for note in r.get("notes", []):
            st.caption(note)

    if r.get("code"):
        with st.expander("View generated code"):
            st.code(r["code"], language="python")
            if r.get("corrections"):
                st.caption(f"The code was corrected {r['corrections']} time(s) after an error.")
