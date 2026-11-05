# واجهة Streamlit على شكل شات: رفع الملف بالـ sidebar، والأسئلة والأجوبة بالشات


import hashlib
import os

import streamlit as st

import config
import ui
from data_quality import REJECTED, load_and_clean
from full_system import ask_agent

st.set_page_config(page_title="Talk to CSV", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
ui.apply_style()

# حالة الجلسة: الرسائل، والسؤال التوضيحي اللي لسا مستنيين جوابه
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("use_sample", False)


def reset_chat():
    st.session_state.messages = []
    st.session_state.pending = None


# ---------------------------------------------------------------
# 1) الـ sidebar: رفع الملف
# ---------------------------------------------------------------
with st.sidebar:
    ui.sidebar_brand()
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if st.button("Use the sample dataset", width="stretch", key="sample_sidebar"):
        st.session_state.use_sample = True
        st.rerun()

# من وين الملف؟ (رفع المستخدم أو الملف التجريبي)
if uploaded is not None:
    st.session_state.use_sample = False
    file_bytes, file_name = uploaded.getvalue(), uploaded.name
elif st.session_state.use_sample:
    with open(os.path.join(os.path.dirname(__file__), config.SAMPLE_FILE), "rb") as f:
        file_bytes, file_name = f.read(), config.SAMPLE_FILE
else:
    # الشاشة الأولى: ما في ملف بعد
    st.markdown('<div class="hero-title">Ask questions about your CSV file</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-text">Upload a file from the sidebar, or try the sample dataset. '
        'The agent writes the Pandas code, checks it for safety, runs it on your data and shows the result. '
        'If your question is unclear, it asks you before answering.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Try the sample dataset", type="primary", key="sample_main"):
        st.session_state.use_sample = True
        st.rerun()
    st.stop()

# ملف مختلف؟ نعيد فحص الجودة ونبدأ شات جديد
file_key = hashlib.md5(file_bytes).hexdigest() + file_name
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.report = load_and_clean(file_bytes, file_name)
    reset_chat()

report = st.session_state.report

# ---------------------------------------------------------------
# 2) الـ sidebar: ملخص الداتا
# ---------------------------------------------------------------
with st.sidebar:
    st.divider()
    ui.sidebar_dataset(file_name, report)
    if report.cleaned_df is not None:
        if st.button("Dataset details", width="stretch", key="details"):
            ui.show_dataset_details(report)
        if st.button("New chat", width="stretch", key="new_chat"):
            reset_chat()
            st.rerun()

# ---------------------------------------------------------------
# 3) الملف مرفوض: نعرض السبب وما في شات
# ---------------------------------------------------------------
if report.cleaned_df is None:
    ui.page_header("This file can't be analyzed")
    st.error("The dataset was rejected. Fix the problems below and upload it again.")
    st.markdown(report.to_markdown())
    if report.original_df is not None:
        st.caption("First 5 rows of the original file")
        st.dataframe(report.original_df.head(), width="stretch")
    st.stop()

cleaned = report.cleaned_df
ui.page_header(f"{file_name}, {report.cleaned_rows:,} rows and {report.cleaned_cols} columns")

# ---------------------------------------------------------------
# 4) الشات: عرض الرسائل السابقة
# ---------------------------------------------------------------
messages = st.session_state.messages
pending = st.session_state.pending

if not messages:
    with st.chat_message("assistant", avatar=":material/analytics:"):
        st.markdown(f"I loaded **{file_name}**. Ask me anything about it.")
    ui.starter_buttons(cleaned)

for i, m in enumerate(messages):
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["text"])
    else:
        with st.chat_message("assistant", avatar=":material/analytics:"):
            ui.render_assistant(m["result"], i, is_last=(i == len(messages) - 1), pending=pending)

# ---------------------------------------------------------------
# 5) سؤال جديد (من الكتابة أو من زر مقترح)
# ---------------------------------------------------------------
questions_asked = sum(1 for m in messages if m["role"] == "user")
limit_reached = questions_asked >= config.MAX_QUESTIONS_PER_SESSION

queued = st.session_state.pop("queued", None)
typed = st.chat_input("Ask a question about your data", max_chars=config.MAX_QUESTION_LENGTH,
                      disabled=limit_reached)
prompt = (queued or typed or "").strip()

if limit_reached:
    st.info("You reached the question limit for this session. Click New chat to continue.")

if prompt and not limit_reached:
    messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # لو النظام كان سأل سؤال توضيح، هاد الكلام هو الجواب عليه
    if pending:
        original = pending["question"]
        clarifications = pending["clarifications"] + [{"question": pending["asked"], "answer": prompt}]
    else:
        original, clarifications = prompt, []

    with st.chat_message("assistant", avatar=":material/analytics:"):
        with st.spinner("Analyzing your data..."):
            try:
                result = ask_agent(original, cleaned, report, clarifications)
            except Exception as e:  # أي خطأ غير متوقع ما لازم يطيّح التطبيق
                result = {"question": original, "action": "error", "summary": "Unexpected error",
                          "message": f"Something went wrong while analyzing: {e}"}

    if result["action"] == "ask_clarification":
        st.session_state.pending = {"question": original, "clarifications": clarifications,
                                    "asked": result["message"]}
    else:
        st.session_state.pending = None

    messages.append({"role": "assistant", "result": result})
    st.rerun()
