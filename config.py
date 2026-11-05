
# كل الإعدادات في مكان واحد عشان يسهل تعديلها
import os
from dotenv import load_dotenv

# نقرأ ملف .env 
load_dotenv()

# ---------------------------------------------------------------
# إعدادات الـ LLM (أي مزود بيدعم نفس صيغة OpenAI)
# ---------------------------------------------------------------
# بنقرأ الإعداد من متغيرات البيئة (.env) وإذا ما لقيناه، من Streamlit Secrets (وقت النشر)
def _get(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


# المزودات الجاهزة: الرابط + المودل الافتراضي + اسم متغير المفتاح
# (كلهم عندهم خطة مجانية بدون بطاقة؛ تأكد من اسم المودل من لوحة المزود لأنه ممكن يتغير)
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.1-flash-lite",
        "key_name": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "key_name": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key_name": "OPENROUTER_API_KEY",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "Qwen/Qwen3-Coder-Next:novita",
        "key_name": "HF_API_KEY",
    },
}

LLM_PROVIDER = (_get("LLM_PROVIDER", "gemini") or "gemini").strip().lower()
if LLM_PROVIDER not in PROVIDERS:
    LLM_PROVIDER = "gemini"
_provider = PROVIDERS[LLM_PROVIDER]

BASE_URL = _get("LLM_BASE_URL") or _provider["base_url"]
MODEL_NAME = _get("LLM_MODEL") or _get("HF_MODEL") or _provider["model"]
KEY_NAME = _provider["key_name"]
LLM_API_KEY = _get(KEY_NAME) or _get("LLM_API_KEY")

# ---------------------------------------------------------------
# عتبات القيم المفقودة (Missing values)
# ---------------------------------------------------------------
WARNING_MISSING_RATE = 0.10          # فوق 10% -> تحذير أوضح
HIGH_MISSING_RATE = 0.30             # فوق 30% -> عمود high-missingness
UNRELIABLE_COLUMN_RATE = 0.50        # فوق 50% -> عمود غير موثوق
MAX_DATASET_MISSING_RATE = 0.40      # فوق 40% من كل الداتا -> نرفض الملف

# ---------------------------------------------------------------
# عتبات الصفوف المكررة (تطابق تام فقط)
# ---------------------------------------------------------------
DUPLICATE_WARNING_RATE = 0.10        # فوق 10% -> نحذف مع تحذير
DUPLICATE_REJECT_RATE = 0.20         # فوق 20% -> نرفض الملف

# ---------------------------------------------------------------
# تحويل أنواع البيانات
# ---------------------------------------------------------------
TYPE_CONVERSION_THRESHOLD = 0.90     # لازم 90% من القيم تتحول عشان نحول العمود
SUSPICIOUS_MIN_RATE = 0.20           # بين 20% و 90% -> عمود مختلط، نحذر ولا نحول

# ---------------------------------------------------------------
# إعدادات عامة
# ---------------------------------------------------------------
MAX_CODE_RETRIES = 2                 # أقصى عدد محاولات تصحيح للكود
NATURAL_LANGUAGE_ANSWERS = True      # جواب نصي طبيعي (بيستهلك طلب LLM إضافي لكل سؤال)
MAX_CLARIFICATIONS = 2               # أقصى عدد أسئلة توضيح للسؤال الواحد
MAX_QUESTIONS_PER_SESSION = 40       # حد للأسئلة بالجلسة الواحدة (يحمي مفتاح الـ API بعد النشر)
MAX_FILE_MB = 20                     # أقصى حجم للملف
SAMPLE_ROWS_FOR_LLM = 3              # عدد الصفوف اللي نرسلها كعينة للـ LLM
MAX_QUESTION_LENGTH = 500
SAMPLE_FILE = "sample_data.csv"
