"""Business logic. Keep Streamlit code OUT of this file so every function can be
tested on its own and reused by the dashboard AND the assistant.

Sections
  1. loading helpers (legacy) and profiling
  2. model layer: join_tables, add_revenue, add_date_parts, column detectors
  3. the tool contract (ToolResult)
  4. analytical tools: get_total, group_total, monthly_revenue, top_n,
     revenue_share, yoy_growth, data_quality_report
"""
from __future__ import annotations

import io
import re
import warnings
from dataclasses import asdict, dataclass, field, replace

import pandas as pd

EXCEL_EXT = ('.xlsx', '.xlsm', '.xls')
TEXT_EXT = ('.csv', '.txt', '.tsv')
BLANK = '(blank)'  # label used in filters for rows where the value is missing


# ================================================================ 1. loading / profiling
def excel_sheet_names(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def load_table(file_bytes: bytes, filename: str, sheet_name=0) -> pd.DataFrame:
    """Read an uploaded CSV / TXT / TSV / Excel file into a DataFrame."""
    name = filename.lower()
    buffer = io.BytesIO(file_bytes)
    if name.endswith(EXCEL_EXT):
        return pd.read_excel(buffer, sheet_name=sheet_name)
    if name.endswith(TEXT_EXT):
        return pd.read_csv(buffer, sep=None, engine='python')  # sniffs the delimiter
    raise ValueError(f'Unsupported file type: {filename}')


def profile_table(df: pd.DataFrame) -> dict:
    """Basic profile: size, duplicates, missing values, data types."""
    missing = df.isna().sum()
    return {
        'rows': len(df),
        'columns': df.shape[1],
        'duplicate_rows': int(df.duplicated().sum()),
        'missing': pd.DataFrame({
            'column': missing.index,
            'missing': missing.values,
            'pct_missing': (missing.values / max(len(df), 1) * 100).round(2),
        }),
        'dtypes': pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values}),
    }


# ================================================================ 2. model layer
class JoinError(ValueError):
    """A join was refused because it would be unsafe. The message is safe to show."""


def _is_texty(s: pd.Series) -> bool:
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype))


def _norm_key(s: pd.Series) -> pd.Series:
    """Make join keys comparable: trim spaces, treat '' as missing, and make 101 == 101.0.
    Only used for matching. The real columns are never changed."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        def fmt(v):
            if pd.isna(v):
                return None
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        return s.map(fmt).astype(object)
    out = s.astype('string').str.strip()
    out = out.mask(out == '')
    return out.astype(object).where(out.notna(), None)


def join_tables(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str,
                right_name: str = 'lookup', value_col: str | None = None):
    """Safe many-to-one LEFT join: every row of `left` is kept exactly once.

    Refuses (JoinError) if `right_on` is not unique, because that would duplicate rows and
    inflate totals. Returns (merged, report). The report shows unmatched rows, so a key
    problem (spaces, text vs number, a missing store) is visible instead of silent.
    Right-hand columns whose names already exist are renamed 'name (right_name)'.
    The right-hand key column is dropped (it equals the left key)."""
    for frame, col, side in ((left, left_on, 'the model so far'), (right, right_on, right_name)):
        if col not in frame.columns:
            raise JoinError(f"Column '{col}' was not found in {side}.")

    lk = _norm_key(left[left_on])
    rk = _norm_key(right[right_on])
    keep = rk.notna().values
    right_ok, rk_ok = right.loc[keep], rk[keep]

    dup = rk_ok[rk_ok.duplicated(keep=False)]
    if len(dup):
        examples = sorted(map(str, dup.unique()))[:5]
        raise JoinError(
            f"'{right_on}' in {right_name} is not unique ({dup.nunique()} value(s) repeat, "
            f"e.g. {examples}). Joining on it would duplicate rows and inflate totals. "
            f"Choose a column that identifies each row once.")

    rest = right_ok.drop(columns=[right_on])
    rest = rest.rename(columns={c: f'{c} ({right_name})' for c in rest.columns if c in left.columns})
    added = list(rest.columns)
    rest = rest.assign(_join_key=rk_ok.values)
    merged = left.assign(_join_key=lk.values).merge(
        rest, on='_join_key', how='left', validate='many_to_one', indicator=True)

    unmatched = merged['_merge'] == 'left_only'
    n_unmatched = int(unmatched.sum())
    report = {
        'right_table': right_name, 'left_key': left_on, 'right_key': right_on,
        'rows_before': len(left), 'rows_after': len(merged),
        'row_count_changed': len(merged) != len(left),
        'unmatched_rows': n_unmatched,
        'unmatched_pct': round(100 * n_unmatched / max(len(left), 1), 2),
        'null_key_rows': int(lk.isna().sum()),
        'unmatched_key_examples': [str(k) for k in merged.loc[unmatched, '_join_key'].dropna().unique()[:10]],
        'lookup_rows_never_used': int(len(right_ok) - rk_ok.isin(set(lk.dropna())).sum()),
        'added_columns': added,
    }
    if value_col and value_col in left.columns:
        before = float(pd.to_numeric(left[value_col], errors='coerce').sum())
        after = float(pd.to_numeric(merged[value_col], errors='coerce').sum())
        report.update(sum_before=before, sum_after=after, sum_changed=abs(before - after) > 1e-6)
    return merged.drop(columns=['_join_key', '_merge']), report


def add_revenue(df: pd.DataFrame, qty_col: str, price_col: str,
                discount_col: str | None = None, name: str = 'Revenue'):
    """Create a revenue column = quantity x price (x (1 - discount)).
    discount_col must hold FRACTIONS (0.10 = 10%). Returns (new_df, report)."""
    for col in (qty_col, price_col, discount_col):
        if col and col not in df.columns:
            raise ValueError(f"Column '{col}' was not found.")
    if name in df.columns:
        raise ValueError(f"A column named '{name}' already exists. Use it, or choose another name.")
    qty = pd.to_numeric(df[qty_col], errors='coerce').astype('float64')
    price = pd.to_numeric(df[price_col], errors='coerce').astype('float64')
    revenue = qty * price
    definition = f'{qty_col} x {price_col}'
    if discount_col:
        disc = pd.to_numeric(df[discount_col], errors='coerce').astype('float64')
        if disc.max() > 1 or disc.min() < 0:
            raise ValueError(f"'{discount_col}' has values outside 0-1. Convert percentages to "
                             f"fractions first (10% -> 0.10).")
        revenue = revenue * (1 - disc.fillna(0))
        definition += f' x (1 - {discount_col})'
    out = df.copy()
    out[name] = revenue
    return out, {'definition': definition, 'rows': len(df),
                 'rows_without_revenue': int(revenue.isna().sum()),
                 'total': float(revenue.sum())}


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pd.to_datetime(s, errors='coerce')


def add_date_parts(df: pd.DataFrame, date_col: str):
    """Make `date_col` a real datetime and add '<col> Year' and '<col> Month' (1-12) columns.
    Returns (new_df, report)."""
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' was not found.")
    parsed = _parse_dates(df[date_col])
    out = df.copy()
    out[date_col] = parsed
    year_col, month_col = f'{date_col} Year', f'{date_col} Month'
    out[year_col] = parsed.dt.year.astype('Int64')
    out[month_col] = parsed.dt.month.astype('Int64')
    return out, {
        'year_col': year_col, 'month_col': month_col,
        'unparseable': int(parsed.isna().sum() - df[date_col].isna().sum()),
        'first': None if parsed.dropna().empty else str(parsed.min().date()),
        'last': None if parsed.dropna().empty else str(parsed.max().date()),
    }


# ---- column detectors: they keep the dashboard dropdowns meaningful
_ID_LIKE = re.compile(r'(?i)(^|[\s_.-])id($|[\s_(.-])')


def find_date_columns(df: pd.DataFrame, min_parse: float = 0.9, sample: int = 500) -> list[str]:
    """Columns that are real dates, or text that is (almost) entirely dates."""
    found = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(col)
        elif _is_texty(s):
            v = s.dropna().astype(str).str.strip().head(sample)
            if v.empty or v.str.fullmatch(r'[-+]?\d+(\.\d+)?').mean() > 0.5:
                continue  # empty, or numbers pretending to be dates
            if _parse_dates(v).notna().mean() >= min_parse:
                found.append(col)
    return found


def find_numeric_columns(df: pd.DataFrame, exclude=()) -> list[str]:
    found = []
    for col in df.columns:
        s = df[col]
        if col in exclude or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            found.append(col)
        elif _is_texty(s):
            v = s.dropna().head(500)
            if len(v) and pd.to_numeric(v, errors='coerce').notna().mean() >= 0.99:
                found.append(col)
    return found


def find_category_columns(df: pd.DataFrame, exclude=(), min_unique: int = 2, max_unique: int = 50) -> list[str]:
    """Text columns with a manageable number of distinct values. Identifier columns
    (names containing 'ID') are skipped: they make poor filters and meaningless charts."""
    found = []
    for col in df.columns:
        if col in exclude or _ID_LIKE.search(str(col)) or not _is_texty(df[col]):
            continue
        if min_unique <= df[col].nunique(dropna=True) <= max_unique:
            found.append(col)
    return found


def suggest_revenue_column(df: pd.DataFrame) -> str | None:
    numeric = find_numeric_columns(df)
    for pattern in ('revenue', 'sales', 'amount', 'total'):
        for col in numeric:
            if pattern in str(col).lower():
                return col
    return None


def suggest_join_keys(left: pd.DataFrame, right: pd.DataFrame, min_overlap: float = 0.5) -> list[dict]:
    """Rank (left column, right column) pairs that could be a many-to-one key.
    The right column must be unique. Overlap = share of left rows whose key exists on the right."""
    def squash(name):
        return re.sub(r'[^a-z0-9]', '', str(name).lower())

    # iterate by POSITION so a table with repeated column names cannot break this, and skip
    # any single column that cannot be read rather than failing the whole app
    right_sets = {}
    for i, rc in enumerate(right.columns):
        try:
            k = _norm_key(right.iloc[:, i]).dropna()
            if len(k) and k.is_unique:
                right_sets[rc] = set(k)
        except Exception:
            continue

    suggestions = []
    for i, lc in enumerate(left.columns):
        try:
            lk = _norm_key(left.iloc[:, i]).dropna()
        except Exception:
            continue
        if lk.empty:
            continue
        for rc, rset in right_sets.items():
            overlap = float(lk.isin(rset).mean())
            same_name = squash(lc) == squash(rc)
            if overlap >= 0.9 or (same_name and overlap >= min_overlap):
                suggestions.append({'left': lc, 'right': rc, 'overlap': round(overlap, 4),
                                    '_score': overlap + (0.05 if same_name else 0)})
    suggestions.sort(key=lambda d: -d['_score'])
    for d in suggestions:
        d.pop('_score')
    return suggestions


# ---- filters shared by the dashboard (so the dashboard and assistant agree)
def filter_options(series: pd.Series) -> list:
    """Sorted distinct values, plus BLANK when some rows are missing this value."""
    vals = series.dropna().unique().tolist()
    try:
        vals = sorted(vals)
    except TypeError:
        vals = sorted(vals, key=str)
    if series.isna().any():
        vals.append(BLANK)
    return vals


def filter_frame(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Keep rows where each column's value is in the selected list.
    BLANK in a selection means 'also keep rows where the value is missing', so rows with
    missing values (for example the Website store has no City) are not silently lost."""
    mask = pd.Series(True, index=df.index)
    for col, selected in selections.items():
        chosen = [v for v in selected if v != BLANK]
        keep = df[col].isin(chosen)
        if BLANK in selected:
            keep = keep | df[col].isna()
        mask &= keep
    return df[mask]


# ================================================================ 3. tool contract
@dataclass
class ToolResult:
    """Every analytical function returns this. The assistant may only narrate it.
    Keep `value` to plain Python types (numbers, strings, dicts, lists)."""
    tool: str
    value: object
    definition: str
    scope: dict
    rows_used: int
    caveats: list = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _error(tool: str, message: str, scope: dict | None = None) -> ToolResult:
    return ToolResult(tool=tool, value=None, definition='', scope=scope or {},
                      rows_used=0, caveats=[], error=message)


def _apply_filters(df: pd.DataFrame, filters: dict | None):
    """Validate filters against the LOADED data, then apply them."""
    for col, wanted in (filters or {}).items():
        if col not in df.columns:
            return df, f'Unknown column: {col}'
        wanted_list = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
        valid = set(df[col].dropna().unique())
        bad = [w for w in wanted_list if w not in valid]
        if bad:
            examples = sorted(map(str, valid))[:20]
            return df, f'{bad} not found in column {col}. Valid values include: {examples}'
        df = df[df[col].isin(wanted_list)]
    return df, None


def _scope(filters):
    return {'filters': filters or 'none (all rows)'}


# ================================================================ 4. analytical tools
def get_total(df: pd.DataFrame, value_col: str = 'Revenue', filters: dict | None = None) -> ToolResult:
    """Sum of one numeric column, optionally filtered."""
    name = 'get_total'
    if value_col not in df.columns:
        return _error(name, f'Unknown column: {value_col}')
    subset, err = _apply_filters(df, filters)
    if err:
        return _error(name, err, {'filters': filters})
    values = pd.to_numeric(subset[value_col], errors='coerce')
    n_bad = int(values.isna().sum())
    caveats = ['This is a sum of the chosen column. Revenue is not profit.']
    if n_bad:
        caveats.append(f'{n_bad} row(s) had a missing or non-numeric {value_col} and were excluded.')
    return ToolResult(tool=name, value=float(values.sum()), definition=f'Sum of {value_col}',
                      scope=_scope(filters), rows_used=int(values.notna().sum()), caveats=caveats)


def group_total(df: pd.DataFrame, value_col: str, group_col: str,
                filters: dict | None = None) -> ToolResult:
    """Sum of one numeric column by group (store type, category, year, ...)."""
    name = 'group_total'
    for col in (value_col, group_col):
        if col not in df.columns:
            return _error(name, f'Unknown column: {col}')
    subset, err = _apply_filters(df, filters)
    if err:
        return _error(name, err, {'filters': filters})
    work = subset.assign(**{value_col: pd.to_numeric(subset[value_col], errors='coerce')})
    n_bad = int(work[value_col].isna().sum())
    n_nogroup = int(work[group_col].isna().sum())
    valid = work.dropna(subset=[value_col, group_col])
    grouped = valid.groupby(group_col)[value_col].sum()
    caveats = ['Sums of the chosen column. Revenue is not profit.']
    if n_bad:
        caveats.append(f'{n_bad} row(s) with missing or non-numeric {value_col} were excluded.')
    if n_nogroup:
        caveats.append(f'{n_nogroup} row(s) with a missing {group_col} were excluded.')
    return ToolResult(tool=name, value={str(k): float(v) for k, v in grouped.items()},
                      definition=f'Sum of {value_col} by {group_col}',
                      scope=_scope(filters), rows_used=len(valid), caveats=caveats)


def monthly_revenue(df: pd.DataFrame, date_col: str, value_col: str = 'Revenue',
                    filters: dict | None = None) -> ToolResult:
    """Sum of a value column per calendar month ('YYYY-MM'). Months with no rows are omitted."""
    name = 'monthly_revenue'
    for col in (date_col, value_col):
        if col not in df.columns:
            return _error(name, f'Unknown column: {col}')
    subset, err = _apply_filters(df, filters)
    if err:
        return _error(name, err, {'filters': filters})
    work = pd.DataFrame({'d': _parse_dates(subset[date_col]),
                         'v': pd.to_numeric(subset[value_col], errors='coerce').astype('float64')})
    n_bad = int(work.isna().any(axis=1).sum())
    work = work.dropna()
    monthly = work.groupby(work['d'].dt.to_period('M'))['v'].sum()
    caveats = ['Months with no matching rows are not shown. Revenue is not profit.']
    if n_bad:
        caveats.append(f'{n_bad} row(s) with an invalid date or value were excluded.')
    return ToolResult(tool=name, value={str(p): float(t) for p, t in monthly.items()},
                      definition=f'Sum of {value_col} per calendar month of {date_col}',
                      scope=_scope(filters), rows_used=len(work), caveats=caveats)


def top_n(df: pd.DataFrame, value_col: str, group_col: str, n: int = 5,
          filters: dict | None = None) -> ToolResult:
    """The n groups with the largest sum of value_col, largest first."""
    try:
        n = int(n)
        if n < 1:
            raise ValueError
    except (TypeError, ValueError):
        return _error('top_n', 'n must be a whole number of at least 1')
    res = group_total(df, value_col, group_col, filters)
    if res.error:
        return replace(res, tool='top_n')
    ranked = dict(sorted(res.value.items(), key=lambda kv: kv[1], reverse=True)[:n])
    return replace(res, tool='top_n', value=ranked,
                   definition=f'Top {n} {group_col} by sum of {value_col}')


def revenue_share(df: pd.DataFrame, value_col: str, group_col: str,
                  filters: dict | None = None) -> ToolResult:
    """Each group's total and its percent of the overall total, largest first."""
    res = group_total(df, value_col, group_col, filters)
    if res.error:
        return replace(res, tool='revenue_share')
    total = sum(res.value.values())
    if total == 0:
        return _error('revenue_share', 'The total is zero, so shares cannot be calculated.', res.scope)
    value = {k: {'total': v, 'share_pct': round(v / total * 100, 2)}
             for k, v in sorted(res.value.items(), key=lambda kv: kv[1], reverse=True)}
    caveats = list(res.caveats)
    if any(v < 0 for v in res.value.values()):
        caveats.append('Some groups are negative, so shares can exceed 100% or fall below 0%.')
    return replace(res, tool='revenue_share', value=value, caveats=caveats,
                   definition=f'Share of total {value_col} by {group_col} (percent of the groups shown)')


def yoy_growth(df: pd.DataFrame, date_col: str, value_col: str = 'Revenue',
               filters: dict | None = None) -> ToolResult:
    """Yearly totals and the percent change versus the previous calendar year."""
    name = 'yoy_growth'
    for col in (date_col, value_col):
        if col not in df.columns:
            return _error(name, f'Unknown column: {col}')
    subset, err = _apply_filters(df, filters)
    if err:
        return _error(name, err, {'filters': filters})
    work = pd.DataFrame({'d': _parse_dates(subset[date_col]),
                         'v': pd.to_numeric(subset[value_col], errors='coerce').astype('float64')})
    n_bad = int(work.isna().any(axis=1).sum())
    work = work.dropna()
    value, prev_year, prev_total = {}, None, None
    for year, part in work.groupby(work['d'].dt.year):
        total = float(part['v'].sum())
        comparable = prev_year is not None and year == prev_year + 1 and prev_total
        growth = round((total - prev_total) / abs(prev_total) * 100, 2) if comparable else None
        value[str(int(year))] = {'total': total, 'yoy_pct': growth,
                                 'months_with_data': int(part['d'].dt.to_period('M').nunique())}
        prev_year, prev_total = year, total
    caveats = ['Growth compares a year with the one before it. Revenue is not profit.']
    if any(v['months_with_data'] < 12 for v in value.values()):
        caveats.append('Some years have fewer than 12 months of data (or a filter removed months), '
                       'so their totals and growth are not like-for-like.')
    if n_bad:
        caveats.append(f'{n_bad} row(s) with an invalid date or value were excluded.')
    return ToolResult(tool=name, value=value, definition=f'Yearly sum of {value_col} and % change vs prior year',
                      scope=_scope(filters), rows_used=len(work), caveats=caveats)


def data_quality_report(df: pd.DataFrame) -> ToolResult:
    """Row count, duplicate rows, and missing values per column."""
    missing = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    caveats = []
    if missing:
        caveats.append('Missing values may limit conclusions drawn from those columns.')
    return ToolResult(tool='data_quality_report',
                      value={'rows': len(df), 'duplicate_rows': int(df.duplicated().sum()),
                             'columns_with_missing': missing},
                      definition='Counts of duplicate rows and missing values in the loaded table',
                      scope={'filters': 'none (full table)'}, rows_used=len(df), caveats=caveats)


# ------------------------------------------------------------ merge check (legacy helper)
def check_merge(left: pd.DataFrame, right: pd.DataFrame, on, value_col: str, how: str = 'left'):
    """Merge, and report whether the join changed the row count or the revenue sum.
    Prefer join_tables(), which also refuses unsafe joins. Returns (merged, report)."""
    before_rows = len(left)
    before_sum = float(pd.to_numeric(left[value_col], errors='coerce').sum())
    merged = left.merge(right, on=on, how=how)
    after_rows = len(merged)
    after_sum = float(pd.to_numeric(merged[value_col], errors='coerce').sum())
    fan_out = after_rows > before_rows or abs(after_sum - before_sum) > 1e-9
    report = {
        'rows_before': before_rows, 'rows_after': after_rows,
        'sum_before': before_sum, 'sum_after': after_sum,
        'changed': fan_out,
        'warning': ('The join changed the row count or the total. Check the grain and keys '
                    'of both tables before using this result.') if fan_out else '',
    }
    return merged, report