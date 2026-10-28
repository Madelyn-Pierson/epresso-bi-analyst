"""Multi-file loading. Works for ANY dataset: nothing here knows column names.

One uploaded file becomes one or more *tables*:
  CSV / TXT / TSV      -> 1 table
  Excel                -> 1 table per sheet
  Access / SQLite      -> 1 table per user table

Rule: loading never changes your VALUES. The only changes are structural (finding the
real header row, dropping fully blank rows/columns, making duplicate column names unique)
plus turning text columns that are entirely numbers into numeric columns (text columns
with leading zeros, like ZIP codes, stay text). Every change is written to `notes` and
shown in the app. Dates, categories, and cleaning belong to the next stage.

No Streamlit imports, so this file can be unit-tested on its own.
"""
from __future__ import annotations

import csv
import io
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_EXT = ('.csv', '.txt', '.tsv')
EXCEL_EXT = ('.xlsx', '.xlsm', '.xls')
ACCESS_EXT = ('.accdb', '.mdb')
SQLITE_EXT = ('.db', '.sqlite', '.sqlite3')
SUPPORTED_TYPES = [e.lstrip('.') for e in TEXT_EXT + EXCEL_EXT + ACCESS_EXT + SQLITE_EXT]

DELIMITERS = [',', '\t', ';', '|']
HEADER_SCAN_ROWS = 20   # how far down to look for the real header row
HEADER_MIN_FILL = 0.6   # header must be at least this full compared with the widest row


class LoadError(Exception):
    """A file could not be read. The message is safe to show to the user."""


@dataclass
class LoadedTable:
    file: str
    name: str
    df: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    solo: bool = False  # True when the file contains only this one table

    @property
    def label(self) -> str:
        """What the user sees in dropdowns: the file name, plus the sheet/table if there are several."""
        return self.file if self.solo else f'{self.file} › {self.name}'

    @property
    def usable(self) -> bool:
        """Offer it for analysis: it has at least one row and one column."""
        return len(self.df) > 0 and self.df.shape[1] >= 1


@dataclass
class FileLoad:
    filename: str
    tables: list[LoadedTable]
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- public API
def load_file(file_bytes: bytes, filename: str) -> FileLoad:
    """Read one uploaded file into one or more raw tables. Raises LoadError with a readable message."""
    ext = Path(filename).suffix.lower()
    loaders = ((TEXT_EXT, _load_text), (EXCEL_EXT, _load_excel),
               (ACCESS_EXT, _load_access), (SQLITE_EXT, _load_sqlite))
    for exts, loader in loaders:
        if ext in exts:
            try:
                result = loader(file_bytes, filename)
            except LoadError:
                raise
            except Exception as exc:  # any parser failure becomes a readable message
                raise LoadError(f'{type(exc).__name__}: {exc}') from exc
            for table in result.tables:
                table.solo = len(result.tables) == 1
            return result
    raise LoadError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_TYPES)}.")


def describe_tables(tables: list[LoadedTable]) -> pd.DataFrame:
    """One summary row per table, for the 'Files and tables found' view."""
    return pd.DataFrame([
        {'File': t.file, 'Table': t.name, 'Rows': len(t.df), 'Columns': t.df.shape[1],
         'Usable': 'yes' if t.usable else 'no (empty)', 'Notes': '; '.join(t.notes)}
        for t in tables
    ])


# --------------------------------------------------------------------------- text files
def _decode(data: bytes) -> tuple[str, str]:
    for enc in ('utf-8-sig', 'cp1252'):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return data.decode('latin-1'), 'latin-1'


def _detect_delimiter(text: str) -> str:
    """Choose the delimiter that splits the first lines into the same number (>1) of columns.
    Uses csv.reader, so commas inside quoted fields ("Smith, Jo") do not fool it."""
    lines = [ln for ln in text.splitlines()[:30] if ln.strip()]
    best, best_score = ',', (0.0, 0)
    for delim in DELIMITERS:
        if not lines:
            break
        ncols, nlines = Counter(len(r) for r in csv.reader(lines, delimiter=delim)).most_common(1)[0]
        if ncols < 2:
            continue
        score = (nlines / len(lines), ncols)
        if score > best_score:
            best, best_score = delim, score
    return best


def _load_text(file_bytes: bytes, filename: str) -> FileLoad:
    text, enc = _decode(file_bytes)
    if not text.strip():
        raise LoadError('The file is empty.')
    delim = _detect_delimiter(text)
    notes = []
    if delim != ',':
        notes.append(f"delimiter detected: {'tab' if delim == chr(9) else repr(delim)}")
    if enc != 'utf-8-sig':
        notes.append(f'encoding: {enc}')

    # Read everything as text first (so ragged rows and title lines cannot break the parse),
    # then find the header the same way we do for Excel sheets.
    rows = list(csv.reader(io.StringIO(text, newline=''), delimiter=delim))
    width = max(len(r) for r in rows)
    raw = pd.DataFrame([r + [''] * (width - len(r)) for r in rows], dtype=object)
    raw = raw.mask(raw == '', np.nan)

    df, header_notes = _promote_header(raw)
    df, type_notes = _infer_text_types(df)
    notes += header_notes + type_notes
    if df.shape[1] == 1 and len(df):
        notes.append('only one column found; if this should be a table, check the delimiter')
    return FileLoad(filename, [LoadedTable(filename, Path(filename).stem, df, notes)])


def _infer_text_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Turn text columns into numbers only when EVERY non-blank value is a number and none
    has a leading zero (so ZIP codes, phone fragments and codes like '007' stay text)."""
    converted = []
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        present = s.dropna().astype(str).str.strip()
        if present.empty or present.str.match(r'^[-+]?0\d').any():
            continue
        as_num = pd.to_numeric(present, errors='coerce')
        if as_num.notna().all() and not np.isinf(as_num).any():
            df[col] = pd.to_numeric(s.astype(str).str.strip().where(s.notna()), errors='coerce')
            converted.append(str(col))
    notes = []
    if converted:
        shown = ', '.join(converted[:8]) + (f' (+{len(converted) - 8} more)' if len(converted) > 8 else '')
        notes.append(f'read as numbers: {shown}')
    return df, notes


# --------------------------------------------------------------------------- excel
def _load_excel(file_bytes: bytes, filename: str) -> FileLoad:
    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
    if not sheets:
        raise LoadError('The workbook has no sheets.')
    tables = []
    for sheet_name, raw in sheets.items():
        df, notes = _promote_header(raw)
        tables.append(LoadedTable(filename, str(sheet_name), df, notes))
    return FileLoad(filename, tables)


def _promote_header(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Find the real header row in a table read WITHOUT a header (Excel sheet or text file).

    Files often start with a title or blank rows. The header is the first row that is nearly
    as full as the widest row and contains only text."""
    raw = raw.dropna(how='all').dropna(axis=1, how='all')
    if raw.empty:
        return pd.DataFrame(), ['empty']

    scan = raw.head(HEADER_SCAN_ROWS)
    filled = scan.notna().sum(axis=1)
    width = filled.max()
    header_pos = 0
    for pos in range(len(scan)):
        values = scan.iloc[pos].dropna()
        if filled.iloc[pos] >= HEADER_MIN_FILL * width and all(isinstance(v, str) for v in values):
            header_pos = pos
            break

    notes = []
    header_line = int(raw.index[header_pos]) + 1  # raw.index still holds original row positions
    if header_line > 1:
        notes.append(f'header found on row {header_line} (rows above it ignored)')
    body = raw.iloc[header_pos + 1:].copy()
    body.columns = _clean_columns(raw.iloc[header_pos])
    body = body.reset_index(drop=True).infer_objects()
    if body.empty:
        notes.append('no data rows below the header')
    return body, notes


# --------------------------------------------------------------------------- access
_ACCESS_HIDDEN = re.compile(r'^(MSys|~)|^f_[0-9A-Fa-f]{32}_Data$')  # Access internal tables


def _load_access(file_bytes: bytes, filename: str) -> FileLoad:
    try:
        from access_parser import AccessParser
    except ImportError as exc:
        raise LoadError("Reading Access files needs the 'access-parser' package "
                        '(pip install access-parser).') from exc
    tables, warnings = [], []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'database.accdb'
        path.write_bytes(file_bytes)
        db = AccessParser(str(path))
        for table_name in db.catalog:
            if _ACCESS_HIDDEN.search(table_name):
                continue
            try:
                df = pd.DataFrame(db.parse_table(table_name))
            except Exception as exc:
                warnings.append(f"table '{table_name}' could not be read ({type(exc).__name__}: {exc})")
                continue
            df, notes = _tidy_structure(df)
            tables.append(LoadedTable(filename, table_name, df, notes))
    if not tables:
        raise LoadError('No readable user tables were found in this Access file.')
    return FileLoad(filename, tables, warnings)


# --------------------------------------------------------------------------- sqlite
def _load_sqlite(file_bytes: bytes, filename: str) -> FileLoad:
    if not file_bytes.startswith(b'SQLite format 3\x00'):
        raise LoadError('This is not a SQLite database file.')
    tables = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'database.sqlite'
        path.write_bytes(file_bytes)
        con = sqlite3.connect(str(path))
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for name in names:
                quoted = '"' + name.replace('"', '""') + '"'
                df, notes = _tidy_structure(pd.read_sql_query(f'SELECT * FROM {quoted}', con))
                tables.append(LoadedTable(filename, name, df, notes))
        finally:
            con.close()
    if not tables:
        raise LoadError('No tables were found in this SQLite file.')
    return FileLoad(filename, tables)


# --------------------------------------------------------------------------- shared helpers
def _clean_columns(names) -> list[str]:
    """Strip names, name blank/'Unnamed' columns, and make duplicates unique (x, x.1, x.2)."""
    out, seen = [], Counter()
    for i, n in enumerate(names):
        s = '' if pd.isna(n) else str(n).strip()
        if not s or s.startswith('Unnamed:'):
            s = f'column_{i + 1}'
        seen[s] += 1
        out.append(s if seen[s] == 1 else f'{s}.{seen[s] - 1}')
    return out


def _tidy_structure(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Structural clean-up for tables that already have headers (Access, SQLite)."""
    notes = []
    n_cols = df.shape[1]
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < n_cols:
        notes.append(f'dropped {n_cols - df.shape[1]} fully blank column(s)')
    n_rows = len(df)
    df = df.dropna(how='all')
    if len(df) < n_rows:
        notes.append(f'dropped {n_rows - len(df)} fully blank row(s)')
    cols = _clean_columns(df.columns)
    if cols != [str(c) for c in df.columns]:
        notes.append('column names were trimmed or made unique')
    df.columns = cols
    return df.reset_index(drop=True), notes