"""Tests for loading.py. Put this file next to test_analytics.py.
Synthetic files only, so nothing here depends on a particular dataset. The last section
smoke-tests whatever files you drop into a data/ folder (and is skipped if there are none)."""
import io
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):          # works whether this sits in the project root or a tests/ folder
    sys.path.insert(0, str(p))
import loading  # noqa: E402


def excel_bytes(sheets: dict) -> bytes:
    """sheets: name -> list of rows, written exactly as given (no header handling)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(xw, sheet_name=name, header=False, index=False)
    return buf.getvalue()


def sqlite_bytes(tables: dict) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'x.db'
        con = sqlite3.connect(path)
        for name, df in tables.items():
            df.to_sql(name, con, index=False)
        con.close()
        return path.read_bytes()


def one(raw: bytes, name: str) -> loading.LoadedTable:
    return loading.load_file(raw, name).tables[0]


# ---------------------------------------------------------------- text files
def test_comma_csv_with_numbers_and_text():
    t = one(b'name,qty\nA,1\nB,2\n', 'x.csv')
    assert list(t.df.columns) == ['name', 'qty'] and len(t.df) == 2
    assert pd.api.types.is_numeric_dtype(t.df['qty'])
    assert not pd.api.types.is_numeric_dtype(t.df['name'])


def test_tab_delimited_file_named_csv_with_quoted_commas():
    t = one(b'id\tmanager\nA1\t"Smith, Jo"\nA2\t"Lee, Al"\n', 'stores.csv')
    assert list(t.df.columns) == ['id', 'manager'] and t.df.loc[0, 'manager'] == 'Smith, Jo'
    assert any('tab' in n for n in t.notes)


@pytest.mark.parametrize('delim', [';', '|', '\t'])
def test_other_delimiters(delim):
    t = one(f'a{delim}b{delim}c\n1{delim}2{delim}3\n'.encode(), 'x.txt')
    assert list(t.df.columns) == ['a', 'b', 'c']


def test_leading_zero_codes_stay_text_but_plain_numbers_become_numeric():
    t = one(b'zip,qty,code\n02134,5,007\n10001,6,008\n', 'x.csv')
    assert t.df.loc[0, 'zip'] == '02134' and t.df.loc[0, 'code'] == '007'
    assert pd.api.types.is_numeric_dtype(t.df['qty'])


def test_mixed_column_stays_text_and_blanks_become_missing():
    t = one(b'a,b\n1,x\nabc,\n3,z\n', 'x.csv')
    assert not pd.api.types.is_numeric_dtype(t.df['a'])
    assert t.df['b'].isna().sum() == 1


def test_numeric_column_with_blanks_keeps_blanks_as_nan():
    t = one(b'a,b\n1,x\n,y\n3,z\n', 'x.csv')
    assert t.df['a'].isna().sum() == 1 and t.df['a'].sum() == 4


def test_title_lines_above_header_in_a_csv_are_skipped():
    t = one(b'My Report\n\nid,val\nA,1\nB,2\n', 'x.csv')
    assert list(t.df.columns) == ['id', 'val'] and len(t.df) == 2
    assert any('header found on row 3' in n for n in t.notes)


def test_non_utf8_encoding_is_handled_and_reported():
    t = one('name,city\nJosé,Zürich\n'.encode('cp1252'), 'x.csv')
    assert t.df.loc[0, 'name'] == 'José' and any('encoding' in n for n in t.notes)


def test_duplicate_and_blank_headers_are_fixed():
    t = one(b'a,a,,b\n1,2,x,3\n', 'x.csv')
    assert list(t.df.columns) == ['a', 'a.1', 'column_3', 'b']


def test_single_column_file_is_loaded_but_flagged():
    t = one(b'name\nA\nB\n', 'x.txt')
    assert t.df.shape == (2, 1) and any('one column' in n for n in t.notes)


def test_empty_file_raises_readable_error():
    with pytest.raises(loading.LoadError):
        loading.load_file(b'  \n', 'x.csv')


def test_unsupported_extension_raises():
    with pytest.raises(loading.LoadError, match='Unsupported'):
        loading.load_file(b'hello', 'notes.pdf')


# ---------------------------------------------------------------- excel
def test_excel_title_rows_skipped_and_numbers_stay_numeric():
    rows = [['Report', None, None], [None, None, None],
            ['id', 'year', 'score'], ['A', 'Y1', 4.1], ['B', 'Y1', None]]
    t = one(excel_bytes({'R': rows}), 'w.xlsx')
    assert list(t.df.columns) == ['id', 'year', 'score'] and len(t.df) == 2
    assert pd.api.types.is_float_dtype(t.df['score'])


def test_excel_one_table_per_sheet_and_empty_sheet_is_unusable():
    wb = excel_bytes({'Data': [['x', 'y'], [1, 2], [3, 4]], 'Notes': [['Author', None], ['Purpose', 'Just notes']]})
    result = loading.load_file(wb, 'w.xlsx')
    tables = {t.name: t for t in result.tables}
    assert set(tables) == {'Data', 'Notes'}
    assert tables['Data'].usable and not tables['Notes'].usable


def test_header_on_first_row_is_left_alone():
    t = one(excel_bytes({'S': [['a', 'b'], [1, 2]]}), 'w.xlsx')
    assert list(t.df.columns) == ['a', 'b'] and t.notes == []


# ---------------------------------------------------------------- sqlite
def test_sqlite_tables_are_read():
    raw = sqlite_bytes({'orders': pd.DataFrame({'id': [1, 2], 'amt': [9.5, 3.0]}),
                        'people': pd.DataFrame({'n': ['a', 'b', 'c']})})
    result = loading.load_file(raw, 'shop.db')
    assert {t.name: len(t.df) for t in result.tables} == {'orders': 2, 'people': 3}


def test_fake_db_file_is_rejected():
    with pytest.raises(loading.LoadError, match='SQLite'):
        loading.load_file(b'not a database', 'x.db')


# ---------------------------------------------------------------- labels
def test_label_is_just_the_filename_for_single_table_files_and_includes_sheet_otherwise():
    single = loading.load_file(b'a,b\n1,2\n', 'one.csv').tables[0]
    assert single.label == 'one.csv'
    multi = loading.load_file(excel_bytes({'A': [['x', 'y'], [1, 2]], 'B': [['x', 'y'], [1, 2]]}), 'w.xlsx')
    assert [t.label for t in multi.tables] == ['w.xlsx › A', 'w.xlsx › B']


# ---------------------------------------------------------------- real files (optional, generic)
DATA_FILES = sorted(p for p in (HERE / 'data').glob('*') if p.suffix.lower().lstrip('.') in loading.SUPPORTED_TYPES) \
    if (HERE / 'data').exists() else []


@pytest.mark.skipif(not DATA_FILES, reason='put sample files in a data/ folder to run this')
@pytest.mark.parametrize('path', DATA_FILES, ids=[p.name for p in DATA_FILES])
def test_every_sample_file_loads_at_least_one_usable_table(path):
    result = loading.load_file(path.read_bytes(), path.name)
    assert any(t.usable for t in result.tables)
    for t in result.tables:   # structural sanity for every table
        assert len(set(t.df.columns)) == len(t.df.columns)