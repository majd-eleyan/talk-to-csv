# تنفيذ الكود بعد ما يمر من الفحص
# Prompt restrictions
#         ↓
# AST validation
#         ↓
# Restricted globals
#         ↓
# Execution
#         ↓
# Error handling

import builtins
import pandas as pd
from code_validator import SAFE_BUILTIN_NAMES, validate_code

# error
# الكود تم رفضه قبل التنفيذ
class UnsafeCodeError(Exception):
    """الكود رفضه الـ AST validator"""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))

# هنا الكود اجتاز الفحص لكنه فشل أثناء التنفيذ.
class ExecutionError(Exception):
    """الكود آمن بس فشل أثناء التنفيذ (مثلاً عمود غير موجود)"""

    def __init__(self, message, original=None):
        self.original = original
        super().__init__(message)


def run_pandas_code(df, code):
    # 1) الفحص أولاً - ما في تنفيذ لأي كود ما تم فحصه
    is_safe, errors = validate_code(code)
    if not is_safe:
        raise UnsafeCodeError(errors)

    # 2) البيئة المقيّدة: بس الدوال الآمنة + df + pd
    safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe_globals = {
        "__builtins__": safe_builtins,
        "df": df.copy(),   # نشتغل على نسخة عشان الداتا الأصلية ما تتغير
        "pd": pd,
    }

    # 3) التنفيذ مع التقاط أي خطأ (التطبيق ما بيقع)
    try:
        exec(compile(code, "<llm_code>", "exec"), safe_globals)
    except Exception as e:
        raise ExecutionError(f"{type(e).__name__}: {e}", original=e) from e

    if "result" not in safe_globals:
        raise ExecutionError("The code did not create a variable named `result`.")
    return safe_globals["result"]
