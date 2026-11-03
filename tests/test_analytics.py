"""Run with:  pytest -q
Tiny tables whose answers you can check in your head. If one of these fails, fix the function
BEFORE connecting it to the assistant."""
import pandas as pd
import pytest

import agent
import analytics as a

sales = pd.DataFrame({
    'Sales ID': ['a', 'b', 'c', 'd'],
    'Date': ['2025-01-31', '2025-02-28', '2025-02-28', '2026-01-31'],
    'Store ID': ['S1', 'S1', 'S2', 'S2'],
    'Product ID': ['P1', 'P2', 'P1', 'P2'],
    'Revenue': [10.0, 20.0, 5.0, 40.0],
})
stores = pd.DataFrame({'ID': ['S1', 'S2'], 'Channel': ['Web', 'Store']})
products = pd.DataFrame({'Product ID': ['P1', 'P2'], 'Product': ['Latte', 'Mocha']})


# ------------------------------------------------------------------ joins
def test_join_keeps_row_count_and_total():
    merged, rep = a.join_tables(sales, stores, 'Store ID', 'ID', 'Stores', value_col='Revenue')
    assert len(merged) == len(sales) and not rep['row_count_changed']
    assert rep['unmatched_rows'] == 0 and not rep['sum_changed']
    assert list(merged['Channel']) == ['Web', 'Web', 'Store', 'Store']


def test_join_refuses_duplicate_lookup_keys():
    bad = pd.DataFrame({'ID': ['S1', 'S1', 'S2'], 'Channel': ['Web', 'Web2', 'Store']})
    with pytest.raises(a.JoinError):
        a.join_tables(sales, bad, 'Store ID', 'ID', 'Stores')


def test_join_reports_unmatched_and_tolerates_spaces_and_number_types():
    s = sales.assign(**{'Store ID': ['S1 ', 'S1', 'S2', 'S9']})  # trailing space, unknown store
    merged, rep = a.join_tables(s, stores, 'Store ID', 'ID', 'Stores')
    assert rep['unmatched_rows'] == 1 and rep['unmatched_key_examples'] == ['S9']
    assert len(merged) == len(s)
    nums = a.join_tables(pd.DataFrame({'k': [1, 2]}), pd.DataFrame({'k2': [1.0, 2.0], 'v': ['x', 'y']}), 'k', 'k2')[0]
    assert list(nums['v']) == ['x', 'y']


def test_join_renames_colliding_columns():
    merged, _ = a.join_tables(sales, products.assign(Revenue=0), 'Product ID', 'Product ID', 'Products')
    assert 'Revenue (Products)' in merged.columns and merged['Revenue'].sum() == 75.0


def test_suggest_join_keys_finds_obvious_keys():
    assert a.suggest_join_keys(sales, stores)[0] == {'left': 'Store ID', 'right': 'ID', 'overlap': 1.0}
    assert a.suggest_join_keys(sales, products)[0]['left'] == 'Product ID'


# ------------------------------------------------------------------ revenue / dates
def test_add_revenue_with_discount():
    df = pd.DataFrame({'Qty': [2, 3], 'Price': [10.0, 5.0], 'Disc': [0.1, None]})
    out, rep = a.add_revenue(df, 'Qty', 'Price', 'Disc')
    assert list(out['Revenue']) == [18.0, 15.0] and rep['total'] == 33.0


def test_add_revenue_rejects_percent_discounts():
    df = pd.DataFrame({'Qty': [1], 'Price': [10.0], 'Disc': [10]})
    with pytest.raises(ValueError):
        a.add_revenue(df, 'Qty', 'Price', 'Disc')


def test_date_parts_and_detectors():
    out, rep = a.add_date_parts(sales, 'Date')
    assert list(out['Date Year']) == [2025, 2025, 2025, 2026] and rep['unparseable'] == 0
    assert a.find_date_columns(sales) == ['Date']          # IDs and names are not dates
    assert a.suggest_revenue_column(sales) == 'Revenue'
    assert 'Sales ID' not in a.find_category_columns(sales.assign(x=1))


def test_filter_frame_keeps_blank_rows():
    df = pd.DataFrame({'City': ['A', None, 'B'], 'Revenue': [1, 2, 3]})
    assert len(a.filter_frame(df, {'City': a.filter_options(df['City'])})) == 3  # nothing lost
    assert len(a.filter_frame(df, {'City': ['A']})) == 1
    assert len(a.filter_frame(df, {'City': ['A', a.BLANK]})) == 2


# ------------------------------------------------------------------ analytical tools
@pytest.fixture
def model():
    m, _ = a.join_tables(sales, stores, 'Store ID', 'ID', 'Stores')
    m, _ = a.join_tables(m, products, 'Product ID', 'Product ID', 'Products')
    return a.add_date_parts(m, 'Date')[0]


def test_get_total(model):
    assert a.get_total(model, 'Revenue').value == 75.0
    assert a.get_total(model, 'Revenue', {'Channel': 'Web'}).value == 30.0


def test_group_total(model):
    r = a.group_total(model, 'Revenue', 'Channel')
    assert r.value == {'Store': 45.0, 'Web': 30.0} and r.rows_used == 4


def test_top_n(model):
    assert list(a.top_n(model, 'Revenue', 'Product', n=1).value) == ['Mocha']  # 60 vs 15
    assert a.top_n(model, 'Revenue', 'Product', n=0).error


def test_revenue_share(model):
    r = a.revenue_share(model, 'Revenue', 'Channel')
    assert r.value['Store'] == {'total': 45.0, 'share_pct': 60.0}
    assert sum(v['share_pct'] for v in r.value.values()) == pytest.approx(100.0)


def test_monthly_revenue(model):
    assert a.monthly_revenue(model, 'Date').value == {'2025-01': 10.0, '2025-02': 25.0, '2026-01': 40.0}


def test_yoy_growth(model):
    r = a.yoy_growth(model, 'Date')
    assert r.value['2025']['total'] == 35.0 and r.value['2025']['yoy_pct'] is None
    assert r.value['2026']['yoy_pct'] == pytest.approx(14.29)       # (40-35)/35
    assert any('fewer than 12 months' in c for c in r.caveats)


def test_errors_are_returned_not_raised(model):
    assert a.get_total(model, 'Nope').error
    assert a.group_total(model, 'Revenue', 'Nope').error
    assert a.get_total(model, 'Revenue', {'Channel': 'Moon'}).error   # value not in data
    assert a.monthly_revenue(model, 'Nope').error


def test_missing_and_bad_values_are_reported(model):
    bad = model.copy()
    bad.loc[0, 'Revenue'] = None
    r = a.get_total(bad, 'Revenue')
    assert r.value == 65.0 and r.rows_used == 3 and any('excluded' in c for c in r.caveats)


# ------------------------------------------------------------------ assistant plumbing
def test_dispatcher_rejects_unknown_tools_and_arguments(model):
    log = []
    assert agent.run_tool('drop_table', {}, model, log).error
    assert agent.run_tool('get_total', {'code': 'x'}, model, log).error
    assert len(log) == 2


def test_rules_router_routes_to_tools(model):
    ask = lambda q: agent.answer_question(q, model, value_col='Revenue', date_col='Date')
    assert ask('revenue by channel')['tool_log'][0]['tool'] == 'group_total'
    assert ask('top 1 product by revenue')['tool_log'][0]['result']['value'] == {'Mocha': 60.0}
    assert ask('share of revenue by channel')['tool_log'][0]['tool'] == 'revenue_share'
    assert ask('monthly trend')['tool_log'][0]['tool'] == 'monthly_revenue'
    assert ask('year over year growth')['tool_log'][0]['tool'] == 'yoy_growth'
    assert ask('what is the weather')['findings'] == agent.UNSUPPORTED