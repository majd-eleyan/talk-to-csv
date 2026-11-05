
# الاتصال مع الـ LLM (Gemini / Groq / OpenRouter / Hugging Face)
# إرسال البرومت إلى الـ LLM واستقبال الرد، مع التعامل مع أخطاء الاتصال والـ .
from openai import OpenAI

import config

# تجمع مشاكل المودل تحت نوع واحد بدل أن ترمي للمستخدم أخطاء مختلفة من
class LLMError(Exception):
    """أي مشكلة بالاتصال مع الـ LLM (مفتاح ناقص، إنترنت، رد فاضي)"""

# رسالة واضحة حسب نوع الخطأ (كود الـ HTTP)
def _friendly_error(e):
    code = getattr(e, "status_code", None)
    provider = config.LLM_PROVIDER
    if code == 429:
        return ("The free limit of the AI provider was reached (too many requests or too many tokens). "
                "Wait a minute and try again.")
    if code == 402:
        return (f"The {provider} account has no credits. Switch to a free provider "
                "by setting LLM_PROVIDER (see the README).")
    if code in (401, 403):
        return f"The {provider} API key was rejected. Check {config.KEY_NAME}."
    if code == 404:
        return (f"The model '{config.MODEL_NAME}' was not found on {provider} (it may be retired). "
                "Run 'python list_models.py' to see the available model names, "
                "then set LLM_MODEL to one of them.")
    return f"Could not get an answer from the {provider} API: {e}"


def ask_llm(system_prompt, user_prompt):
    if not config.LLM_API_KEY:
        raise LLMError(
            f"{config.KEY_NAME} is missing. Create a .env file next to app.py "
            f"with the line: {config.KEY_NAME}=your_api_key_here "
            "(on Streamlit Cloud, add it under Secrets)."
        )

    try:
        # ننشئ الـ client هون (مو فوق) عشان التطبيق ما يقع إذا المفتاح ناقص
        client = OpenAI(base_url=config.BASE_URL, api_key=config.LLM_API_KEY)
        completion = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0, 
        )
    except Exception as e:
        raise LLMError(_friendly_error(e)) from e

    content = completion.choices[0].message.content
    if not content or not content.strip():
        raise LLMError("The AI model returned an empty answer.")
    return content
