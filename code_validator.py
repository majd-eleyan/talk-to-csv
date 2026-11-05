# فحص الكود اللي بيولّده المودل "قبل" التنفيذ
#  AST validation + Whitelist 

import ast

MAX_CODE_LENGTH = 2000

# ---------------------------------------------------------------
# القوائم (Whitelists)
# ---------------------------------------------------------------

# أنواع عقد الـ AST المسموحة فقط (مافي import, loop, lambda, def, class ...)
ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign,
    ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Attribute, ast.Call, ast.keyword, ast.Subscript, ast.Slice,
    ast.Tuple, ast.List, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.BitAnd, ast.BitOr, ast.Invert, ast.USub, ast.UAdd,
    ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)

# دوال بايثون العادية المسموحة (بس هدول بيتم تمريرهم للتنفيذ)
SAFE_BUILTIN_NAMES = {
    "len", "round", "abs", "min", "max", "sum", "sorted",
    "int", "float", "str", "bool", "list",
}

# الأسماء الوحيدة المسموح استخدامها كمتغيرات جاهزة
ALLOWED_NAMES = {"df", "pd", "result"} | SAFE_BUILTIN_NAMES

# دوال/أسماء خطيرة - بنرفضها برسالة واضحة
DANGEROUS_NAMES = {
    "__import__", "open", "eval", "exec", "compile", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint",
    "dir", "type", "object", "help", "exit", "quit",
}

# مكتبات خطيرة (للرسالة فقط، لأن أي import مرفوض أصلاً)
DANGEROUS_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "requests",
    "urllib", "builtins", "importlib",
}

# دوال الكتابة/القراءة من الملفات - ممنوعة
FORBIDDEN_METHODS = {
    "to_csv", "to_excel", "to_json", "to_sql", "to_pickle", "to_parquet",
    "to_clipboard", "to_hdf", "to_feather", "to_html", "to_latex",
    "to_string", "to_markdown", "to_xml", "to_stata",
    "read_csv", "read_excel", "read_json", "read_sql", "read_pickle",
    "read_parquet", "read_html", "read_table", "read_clipboard",
    "eval", "query", "pipe", "apply", "applymap", "map", "transform",
}

# عمليات Pandas المسموحة (كطلب عادي: method call)
ALLOWED_METHODS = {
    # إحصاءات
    "mean", "sum", "count", "min", "max", "median", "std", "var", "nunique",
    "unique", "quantile", "mode", "corr", "cumsum", "diff", "pct_change",
    "idxmax", "idxmin", "size", "abs", "round", "describe", "all", "any",
    # تجميع وترتيب
    "groupby", "agg", "aggregate", "sort_values", "sort_index", "value_counts",
    "nlargest", "nsmallest", "head", "tail", "reset_index", "set_index",
    "pivot_table", "resample", "rename", "assign", "copy", "to_frame",
    # تصفية وقيم مفقودة
    "isna", "isnull", "notna", "notnull", "dropna", "fillna", "isin", "between",
    "drop_duplicates", "drop", "astype",
    # نصوص وتواريخ (.str و .dt)
    "contains", "lower", "upper", "strip", "startswith", "endswith", "len",
    "strftime", "month_name", "day_name", "to_period",
    # دوال pd المسموحة
    "to_datetime", "to_numeric", "Grouper", "Timestamp",
}

# خصائص مسموحة (بدون قوسين)
ALLOWED_ATTRIBUTES = {
    "str", "dt", "loc", "iloc", "columns", "index", "shape", "dtypes", "empty",
    "year", "month", "day", "dayofweek", "quarter", "hour",
}

# code
#  ↓
# parse
#  ↓
# inspect AST
#  ↓
# check forbidden operations
#  ↓
# check allowed methods/attributes
#  ↓
# return validation result
def validate_code(code):
    """
    بترجع (is_safe, errors)
    is_safe = True إذا الكود آمن. errors = رسائل واضحة عن سبب الرفض
    """
    errors = []

    if not isinstance(code, str) or not code.strip():
        return False, ["The generated code is empty."]
    if len(code) > MAX_CODE_LENGTH:
        return False, [f"The generated code is too long (max {MAX_CODE_LENGTH} characters)."]

    # الخطوة 1: نحول الكود إلى شجرة AST (بدون تنفيذه!)
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, [f"Syntax error in generated code: {e.msg} (line {e.lineno})."]

    nodes = list(ast.walk(tree))

    # المتغيرات اللي الكود نفسه بيعرّفها (مثل result أو temp)
    assigned_names = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    # الـ Attribute اللي هي اسم دالة تُستدعى (method call)
    called_attributes = {id(n.func) for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    # الخطوة 2: نفحص كل عقدة
    for node in nodes:

        # أي import مرفوض
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in modules if m.split(".")[0] in DANGEROUS_MODULES]
            hint = f" (dangerous module: {', '.join(bad)})" if bad else ""
            errors.append(f"Imports are not allowed{hint}.")
            continue

        # أي نوع syntax مو بالقائمة البيضاء مرفوض
        if not isinstance(node, ALLOWED_NODES):
            errors.append(f"Python syntax '{type(node).__name__}' is not allowed.")
            continue

        # فحص الأسماء
        if isinstance(node, ast.Name):
            if node.id in DANGEROUS_NAMES:
                errors.append(f"Use of '{node.id}' is not allowed.")
            elif node.id.startswith("__"):
                errors.append(f"Dunder name '{node.id}' is not allowed.")
            elif isinstance(node.ctx, ast.Load) and node.id not in ALLOWED_NAMES | assigned_names:
                errors.append(f"Unknown name '{node.id}'. Only df and pd are available.")

        # فحص الخصائص والدوال (df.xxx)
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                errors.append(f"Access to '{attr}' (dunder/private attribute) is not allowed.")
            elif attr in FORBIDDEN_METHODS and attr.startswith(("to_", "read_")):
                errors.append(f"'{attr}' reads/writes files and is not allowed.")
            elif id(node) in called_attributes:
                if attr not in ALLOWED_METHODS:
                    errors.append(f"Method '{attr}' is not in the allowed Pandas whitelist.")
            elif attr not in ALLOWED_ATTRIBUTES:
                errors.append(f"Attribute '{attr}' is not allowed (use df['column'] for columns).")

        # فحص استدعاء الدوال
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTIN_NAMES:
                    errors.append(f"Function '{node.func.id}' is not allowed.")
            elif not isinstance(node.func, ast.Attribute):
                errors.append("Calling this kind of expression is not allowed.")
            # inplace=True ممنوع لأنه بيعدّل الداتا نفسها
            for kw in node.keywords:
                if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value:
                    errors.append("inplace=True is not allowed.")

        # فحص الإسناد: بس لمتغير أو df['col']
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                is_name = isinstance(target, ast.Name)
                is_df_col = (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "df"
                )
                if not (is_name or is_df_col):
                    errors.append("Assignment is only allowed to a variable name or df['column'].")

    # لازم الكود يخزّن الجواب النهائي بالمتغير result
    if "result" not in assigned_names:
        errors.append("The code must store the final answer in a variable named `result`.")

    unique_errors = list(dict.fromkeys(errors))  # نحذف التكرار ونحافظ على الترتيب
    return len(unique_errors) == 0, unique_errors
