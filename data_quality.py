# طبقة جودة البيانات: بتشتغل "قبل" الـ LLM
# كل القرارات هون deterministic (قواعد ثابتة بـ Pandas) - الـ LLM ما بيقرر شي بالتنظيف

import csv
import io
import warnings
from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

import config

# الحالات الثلاث الممكنة للداتا
READY = "READY"
READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
REJECTED = "REJECTED"


@dataclass
class QualityReport:
    """كل نتائج الفحص بمكان واحد (بنعرضها للمستخدم وبنرسل ملخصها للـ LLM)"""

    status: str = REJECTED
    original_df: pd.DataFrame = None   # النسخة الأصلية: ما بنعدل عليها أبداً
    cleaned_df: pd.DataFrame = None    # النسخة النظيفة: التحليل بيتم عليها
    original_rows: int = 0
    original_cols: int = 0
    cleaned_rows: int = 0
    cleaned_cols: int = 0
    duplicate_rows: int = 0
    duplicate_rate: float = 0.0
    overall_missing_rate: float = 0.0
    removed_empty_columns: list = field(default_factory=list)
    conversions: list = field(default_factory=list)      # (العمود, النوع القديم, النوع الجديد)
    missing: dict = field(default_factory=dict)          # العمود -> (العدد, النسبة)
    high_missing_columns: list = field(default_factory=list)
    unreliable_columns: list = field(default_factory=list)
    mixed_type_columns: list = field(default_factory=list)
    cleaning_actions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    # -----------------------------------------------------------
    def to_markdown(self):
        """التقرير النصي اللي بيظهر للمستخدم"""
        lines = ["**Dataset Quality Report**", ""]
        lines.append(f"- Rows: {self.original_rows} → {self.cleaned_rows} after cleaning")
        lines.append(f"- Columns: {self.original_cols} → {self.cleaned_cols} after cleaning")
        lines.append(f"- Missing values: {self.overall_missing_rate:.1%}")
        lines.append(f"- Duplicate rows: {self.duplicate_rows} ({self.duplicate_rate:.1%})")
        if self.conversions:
            lines += ["", "**Data type conversions**"]
            lines += [f"- ✓ {c}: {old} → {new}" for c, old, new in self.conversions]
        if self.cleaning_actions:
            lines += ["", "**Cleaning actions**"]
            lines += [f"- ✓ {a}" for a in self.cleaning_actions]
        if self.warnings:
            lines += ["", "**Warnings**"]
            lines += [f"- ⚠ {w}" for w in self.warnings]
        if self.errors:
            lines += ["", "**Problems**"]
            lines += [f"- ✖ {e}" for e in self.errors]
        lines += ["", f"**Status: {self.status}**"]
        return "\n".join(lines)

    # -----------------------------------------------------------
    def summary_for_llm(self):
        """ملخص قصير بنرسله للـ LLM عشان يعرف شو تغيّر بالداتا"""
        lines = [
            f"Dataset Status: {self.status}",
            f"Original Rows: {self.original_rows}",
            f"Cleaned Rows: {self.cleaned_rows}",
            f"Original Columns: {self.original_cols}",
            f"Cleaned Columns: {self.cleaned_cols}",
            f"Removed Duplicate Rows: {self.duplicate_rows}",
        ]
        if self.removed_empty_columns:
            lines.append(f"Removed Empty Columns: {', '.join(self.removed_empty_columns)}")
        if self.conversions:
            lines.append("Converted Columns:")
            lines += [f"- {c}: {old} -> {new}" for c, old, new in self.conversions]
        remaining = {c: n for c, (n, _) in self.missing.items() if n > 0}
        if remaining:
            lines.append("Remaining Missing Values:")
            lines += [f"- {c}: {n}" for c, n in remaining.items()]
        if self.unreliable_columns:
            lines.append(
                "Unreliable columns (over 50% missing, avoid unless the user asks): "
                + ", ".join(self.unreliable_columns)
            )
        if self.mixed_type_columns:
            lines.append(
                "Mixed-type columns (not converted, values may be inconsistent): "
                + ", ".join(self.mixed_type_columns)
            )
        return "\n".join(lines)


# ---------------------------------------------------------------
# 1) قراءة الملف والتحقق منه
# ---------------------------------------------------------------
def read_csv_safely(file_bytes, filename):
    """بترجع (DataFrame, قائمة أخطاء). ما بترمي exception للمستخدم أبداً"""
    if not str(filename).lower().endswith(".csv"):
        return None, ["Unsupported file type. Please upload a .csv file."]

    if not file_bytes or not file_bytes.strip():
        return None, ["The file is empty."]

    if len(file_bytes) > config.MAX_FILE_MB * 1024 * 1024:
        return None, [f"The file is larger than {config.MAX_FILE_MB} MB."]

    # نجرب utf-8 أول، وإذا فشل نستخدم latin-1 (ما بتفشل أبداً)
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    # نقرأ سطر العناوين بنفسنا لأن Pandas بيغيّر أسماء الأعمدة المكررة بصمت
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error):
        return None, ["The file has no header row / columns."]

    names = [h.strip() for h in header]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        return None, [
            "Duplicate column names found: "
            + ", ".join(f'"{d}"' for d in duplicates)
            + ". Please rename them in your file and upload it again."
        ]

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        return None, ["The file is empty."]
    except Exception as e:  # ملف تالف أو شكله مو CSV
        return None, [f"The file could not be read as a CSV ({type(e).__name__}: {e})."]

    df.columns = [str(c).strip() for c in df.columns]
    return df, []


# ---------------------------------------------------------------
# 2) تحويل الأنواع بشكل آمن
# ---------------------------------------------------------------
def _try_convert_column(series):
    """
    بنحاول نحول عمود نصي إلى رقم أو تاريخ.
    بترجع (السلسلة الجديدة أو None, نوع التحويل, نسبة النجاح المرصودة)
    """
    non_null = series.notna().sum()
    if non_null == 0:
        return None, None, 0.0

    # أولاً: رقم
    numeric = pd.to_numeric(series, errors="coerce")
    num_rate = numeric.notna().sum() / non_null
    if num_rate >= config.TYPE_CONVERSION_THRESHOLD:
        return numeric, "numeric", num_rate

    # ثانياً: تاريخ
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(series, errors="coerce", format="mixed")
    date_rate = dates.notna().sum() / non_null
    if date_rate >= config.TYPE_CONVERSION_THRESHOLD:
        return dates, "datetime", date_rate

    # ما نجح التحويل: نرجع أعلى نسبة نجاح عشان نحذر إذا كان العمود "مختلط"
    return None, None, max(num_rate, date_rate)


# ---------------------------------------------------------------
# 3) الفحص والتنظيف الرئيسي
# ---------------------------------------------------------------
def run_quality_check(original_df):
    report = QualityReport(original_df=original_df)
    report.original_rows, report.original_cols = original_df.shape

    if original_df.shape[1] == 0:
        report.errors.append("The dataset has no columns.")
        return report
    if original_df.shape[0] == 0:
        report.errors.append("The dataset has no rows.")
        return report

    work = original_df.copy()  # نشتغل على نسخة، الأصلية ما بتتغير

    # --- الصفوف المكررة (تطابق تام فقط) ---
    report.duplicate_rows = int(work.duplicated().sum())
    report.duplicate_rate = report.duplicate_rows / len(work)
    if report.duplicate_rate > config.DUPLICATE_REJECT_RATE:
        report.errors.append(
            f"{report.duplicate_rate:.0%} of the rows are exact duplicates "
            f"(limit is {config.DUPLICATE_REJECT_RATE:.0%}). Please review the file."
        )
    elif report.duplicate_rows > 0:
        report.cleaning_actions.append(
            f"Removed {report.duplicate_rows} exact duplicate rows ({report.duplicate_rate:.1%})"
        )
        if report.duplicate_rate > config.DUPLICATE_WARNING_RATE:
            report.warnings.append(
                f"Duplicate rate is high ({report.duplicate_rate:.1%}); duplicates were removed."
            )
    work = work.drop_duplicates().reset_index(drop=True)

    # --- الأعمدة الفاضية بالكامل ---
    empty_cols = [c for c in work.columns if work[c].isna().all()]
    if empty_cols:
        work = work.drop(columns=empty_cols)
        report.removed_empty_columns = empty_cols
        report.cleaning_actions.append(
            "Removed empty columns: " + ", ".join(empty_cols)
        )
    if work.shape[1] == 0:
        report.errors.append("All columns are empty.")

    # --- تحويل الأنواع ---
    for col in list(work.columns):
        series = work[col]
        if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or is_bool_dtype(series):
            continue  # النوع سليم أصلاً

        converted, kind, rate = _try_convert_column(series)
        if converted is not None:
            old_type = str(series.dtype)
            lost = int(converted.isna().sum() - series.isna().sum())
            work[col] = converted
            report.conversions.append((col, old_type, str(converted.dtype)))
            report.cleaning_actions.append(f"Converted {col} to {kind}")
            if lost > 0:
                report.warnings.append(
                    f"{col}: {lost} value(s) could not be converted and became missing."
                )
        elif rate >= config.SUSPICIOUS_MIN_RATE:
            # عمود مختلط: ما نحوله بشكل أعمى، بس نعلّم المشكلة
            report.mixed_type_columns.append(col)
            report.warnings.append(
                f"{col}: only {rate:.0%} of values look numeric/date, "
                "so it was NOT converted (mixed types)."
            )

    # --- القيم المفقودة (بعد التحويل) ---
    for col in work.columns:
        count = int(work[col].isna().sum())
        rate = count / len(work)
        report.missing[col] = (count, rate)
        if count == 0:
            continue
        if rate > config.UNRELIABLE_COLUMN_RATE:
            report.unreliable_columns.append(col)
            report.warnings.append(f"{col} has {rate:.0%} missing values (unreliable column).")
        elif rate > config.HIGH_MISSING_RATE:
            report.high_missing_columns.append(col)
            report.warnings.append(f"{col} has {rate:.0%} missing values (high missingness).")
        elif rate > config.WARNING_MISSING_RATE:
            report.warnings.append(f"{col} has {rate:.0%} missing values ({count} values).")
        else:
            report.warnings.append(f"{count} missing value(s) in {col} ({rate:.1%}).")

    if work.size > 0:
        report.overall_missing_rate = float(work.isna().sum().sum() / work.size)
    if report.overall_missing_rate > config.MAX_DATASET_MISSING_RATE:
        report.errors.append(
            f"The dataset has {report.overall_missing_rate:.0%} missing values overall "
            f"(limit is {config.MAX_DATASET_MISSING_RATE:.0%})."
        )

    # --- الحالة النهائية ---
    report.cleaned_rows, report.cleaned_cols = work.shape
    if report.errors:
        report.status = REJECTED
        report.cleaned_df = None
    else:
        report.cleaned_df = work
        report.status = READY_WITH_WARNINGS if report.warnings else READY
    return report


# ---------------------------------------------------------------
# 4) الدالة اللي بيستدعيها التطبيق
# ---------------------------------------------------------------
def load_and_clean(file_bytes, filename):
    df, errors = read_csv_safely(file_bytes, filename)
    if errors:
        report = QualityReport()
        report.errors = errors
        return report
    return run_quality_check(df)
