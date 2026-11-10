"""Run with:  python -m pytest -q   (from the repository root)"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402
import analytics as A  # noqa: E402


def _source_files():
    return [(p.name, p.read_bytes()) for p in sorted((ROOT / "data").glob("Support_EX_B_*"))]


def _load(overrides=None):
    files = _source_files()
    tables, _, _ = A.load_tables(files)
    if overrides:
        tables.update(overrides)
    return A.build_dataset(tables)


@pytest.fixture(scope="module")
def ds():
    return _load()


def raw_sales():
    from access_parser import AccessParser
    return pd.DataFrame(AccessParser(str(ROOT / "data" / "Support_EX_B_Sales.accdb")).parse_table("Sales"))


def raw_stores():
    return pd.read_csv(ROOT / "data" / "Support_EX_B_Stores.csv", sep="\t", dtype=str)


def test_original_source_files_load_cleanly():
    tables, msgs, unrec = A.load_tables(_source_files())
    assert set(tables) == {"sales", "stores", "products", "categories", "survey"}
    assert unrec == []  # the workbook's Documentation / Terms sheets are ignored quietly
    assert len(tables["sales"]) == 25200 and len(tables["stores"]) == 21 and len(tables["survey"]) == 105


def test_csv_sales_upload_with_other_tables_from_bundle_gives_same_total(ds):
    sales_csv = raw_sales().to_csv(index=False).encode()
    tables, _, unrec = A.load_tables([("sales.csv", sales_csv)])
    assert list(tables) == ["sales"] and not unrec
    full = {**_load().tables, **tables}
    assert A.total_revenue(A.build_dataset(full).fact) == pytest.approx(A.total_revenue(ds.fact))


# ---- data handling ----
def test_total_matches_raw_sum_and_joins_do_not_duplicate(ds):
    assert A.total_revenue(ds.fact) == pytest.approx(raw_sales()["Revenue"].sum()) == pytest.approx(42174440.0)
    assert len(ds.fact) == len(raw_sales())
    assert all(r["OK"] for r in ds.reconciliation)


def test_channel_year_month_partitions_sum_to_total(ds):
    t = A.total_revenue(ds.fact)
    assert A.revenue_by(ds.fact, "Channel")["Revenue"].sum() == pytest.approx(t)
    assert A.revenue_by_year(ds.fact)["Revenue"].sum() == pytest.approx(t)
    assert A.revenue_by_month(ds.fact)["Revenue"].sum() == pytest.approx(t)
    assert A.revenue_by(ds.fact, "Category")["Revenue"].sum() == pytest.approx(t)


def test_monthly_shares_sum_to_100_per_year(ds):
    ms = A.monthly_share(ds.fact)
    assert ms.groupby("Year")["Share of year %"].sum().round(6).eq(100).all()


def test_product_names_are_not_used_as_join_key(ds):
    assert ds.tables["products"]["Product"].duplicated().any()  # 'Base Model' repeats
    assert ds.fact["Product label"].nunique() == ds.fact["Product ID"].nunique()


def test_filters_recalculate(ds):
    f = A.filter_fact(ds.fact, years=[2028], channels=["Website"])
    assert A.total_revenue(f) == pytest.approx(A.get_total_revenue(ds.fact, year=2028, channel="website")["total_revenue"])
    assert A.total_revenue(A.filter_fact(ds.fact, years=[])) == 0


# ---- $1,000 test ----
def test_adding_1000_changes_total_by_exactly_1000_and_one_bucket():
    base = _load()
    s = base.tables["sales"].copy()
    s["Revenue"] = pd.to_numeric(s["Revenue"])
    s.loc[0, "Revenue"] += 1000
    mod = _load(overrides={"sales": s})
    assert A.total_revenue(mod.fact) - A.total_revenue(base.fact) == pytest.approx(1000)
    for fn in (lambda d: A.revenue_by_year(d.fact), lambda d: A.revenue_by_month(d.fact)):
        diff = (fn(mod)["Revenue"] - fn(base)["Revenue"]).round(6)
        assert (diff != 0).sum() == 1 and diff.sum() == pytest.approx(1000)
    # the assistant's tools read the new data too
    a = agent.run_rules("What is the total revenue?", base, base.fact, "full")["tool_log"][0]["result"]["total_revenue"]
    b = agent.run_rules("What is the total revenue?", mod, mod.fact, "full")["tool_log"][0]["result"]["total_revenue"]
    assert b - a == pytest.approx(1000)


def test_works_with_different_years_and_new_store():
    base = _load()
    s = base.tables["sales"].copy()
    extra = s.head(12).copy()
    extra["Sales ID"] = ["NEW" + str(i) for i in range(12)]
    extra["Date"] = pd.to_datetime(extra["Date"]) + pd.DateOffset(years=5)  # 2030
    stores = pd.concat([base.tables["stores"], pd.DataFrame([{"ID": "SH99", "City": "Austin", "State": "TX", "StoreType": "Brick & Mortar"}])])
    extra["Store ID"] = "SH99"
    mod = _load(overrides={"sales": pd.concat([s, extra]), "stores": stores})
    assert 2030 in set(mod.fact["Year"].dropna())
    assert A.get_total_revenue(mod.fact, store="Austin")["total_revenue"] == pytest.approx(extra["Revenue"].astype(float).sum())


# ---- tools ----
def test_store_trend_no_decline_case(ds):
    only_web = A.get_store_trend(ds.fact, channel="Website")
    assert only_web["stores_compared"] == 1
    grown = ds.fact.copy()
    grown["Revenue"] = grown["Revenue"] * (1 + (grown["Year"] - grown["Year"].min()).astype(float))  # everything grows
    res = A.get_store_trend(grown)
    assert res["stores_that_declined"] == 0 and res["largest_pct_decline"] is None
    assert any("No location declined" in n for n in res["notes"])


def test_unknown_and_ambiguous_inputs_return_errors_not_guesses(ds):
    assert "error" in A.get_total_revenue(ds.fact, year=2035)
    assert "error" in A.get_total_revenue(ds.fact, store="Atlantis")
    amb = A.get_total_revenue(ds.fact, store="Orlando")
    assert "error" in amb and "more than one" in amb["error"]
    assert "error" in A.get_total_revenue(ds.fact, channel="carrier pigeon")


def test_satisfaction_result_reports_missing_and_caveats(ds):
    r = A.get_satisfaction_vs_revenue(ds.fact, ds.survey)
    assert r["location_years_missing_satisfaction"] == 5 and "SH00" in r["locations_missing_all_satisfaction"]
    assert any("causation" in c.lower() for c in r["caveats"])


# ---- dispatcher / agent safety ----
def test_dispatcher_rejects_unknown_tool_and_bad_args(ds):
    log = []
    assert "error" in agent.run_tool("os.system", {"cmd": "ls"}, ds, ds.fact, "s", log)
    assert "error" in agent.run_tool("get_total_revenue", {"evil": 1}, ds, ds.fact, "s", log)
    assert "error" in agent.run_tool("get_total_revenue", {"year": "abc"}, ds, ds.fact, "s", log)
    assert len(log) == 3 and all({"function", "arguments", "result"} <= set(c) for c in log)


def test_no_eval_or_exec_in_code():
    for name in ("agent.py", "analytics.py", "streamlit_app.py"):
        src = (ROOT / name).read_text()
        assert "eval(" not in src.replace("evaluat", "") and "exec(" not in src


@pytest.mark.parametrize("q,tool", [
    ("How much revenue did the website generate in 2028?", "get_total_revenue"),
    ("Which store experienced the greatest percentage revenue decline?", "get_store_trend"),
    ("Which E-Presso store is the most profitable?", "get_data_limitations"),
    ("Does higher customer satisfaction cause higher revenue?", "get_satisfaction_vs_revenue"),
    ("Which sales channel experienced the greatest revenue growth?", "get_revenue_by_channel"),
    ("Which month is busiest?", "get_monthly_revenue_share"),
    ("any missing values or duplicates?", "get_data_quality_report"),
])
def test_rules_routing(ds, q, tool):
    out = agent.run_rules(q, ds, ds.fact, "full")
    assert tool in [c["function"] for c in out["tool_log"]]


def test_rules_unsupported_and_profit_never_claims_profit(ds):
    assert agent.run_rules("What is the weather in Denver?", ds, ds.fact, "full")["tool_log"] == []
    ans = agent.run_rules("Which store is most profitable?", ds, ds.fact, "full")["answer"].lower()
    assert "not profit" in ans or "cannot show profit" in ans


def test_website_2028_matches_independent_calculation(ds):
    s = raw_sales()
    m = s.merge(raw_stores()[["ID", "StoreType"]], left_on="Store ID", right_on="ID")
    expected = m[(m["StoreType"] == "Website") & (m["Date"].astype(str).str[:4] == "2028")]["Revenue"].sum()
    got = agent.run_rules("How much revenue did the website generate in 2028?", ds, ds.fact, "full")["tool_log"][0]["result"]["total_revenue"]
    assert got == pytest.approx(expected)


# ---- LLM loop with a fake Gemini client (no network) ----
class _FakeResp:
    def __init__(self, calls=None, text=None):
        from google.genai import types
        self.function_calls = calls or []
        self.text = text
        parts = [types.Part(function_call=types.FunctionCall(name=c.name, args=c.args)) for c in (calls or [])] or [types.Part(text=text or "")]
        self.candidates = [SimpleNamespace(content=types.Content(role="model", parts=parts))]


class _FakeClient:
    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.seen.append(contents)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_llm_loop_runs_our_function_and_logs_it(ds):
    pytest.importorskip("google.genai")
    call = SimpleNamespace(name="get_total_revenue", args={"year": 2028, "channel": "Website"})
    client = _FakeClient([_FakeResp(calls=[call]), _FakeResp(text="**Findings** The website made $6,339,640 in 2028.")])
    out = agent.answer_question("website 2028?", ds, ds.fact, "full", client=client)
    assert out["mode"] == "llm" and out["tool_log"][0]["function"] == "get_total_revenue"
    assert out["tool_log"][0]["result"]["total_revenue"] == 6339640.0
    # the tool result was passed back to the model
    assert "6339640" in str(client.seen[1][-1])


def test_llm_failure_or_quota_falls_back_to_rules(ds):
    pytest.importorskip("google.genai")
    for err in (RuntimeError("429 RESOURCE_EXHAUSTED quota"), ConnectionError("boom")):
        out = agent.answer_question("total revenue?", ds, ds.fact, "full", client=_FakeClient([err]))
        assert out["mode"] == "rules (fallback)" and out["tool_log"] and "unavailable" in out["notes"][0]


def test_llm_answer_without_tool_is_flagged(ds):
    pytest.importorskip("google.genai")
    client = _FakeClient([_FakeResp(text="Revenue was $1."), _FakeResp(text="Revenue was $1.")])
    out = agent.answer_question("total revenue?", ds, ds.fact, "full", client=client)
    assert any("NOT verified" in n for n in out["notes"])


# ---- the assistant answers exactly what is asked (single item vs list, metric, years) ----
def _store_lines(answer: str) -> int:
    return sum(1 for ln in answer.splitlines() if ln.startswith("- SH"))


def test_singular_question_returns_one_store_not_five(ds):
    out = agent.run_rules("Which store had the greatest revenue decline overall?", ds, ds.fact, "full")
    call = out["tool_log"][0]
    assert call["function"] == "get_store_trend"
    assert call["arguments"]["top_n"] == 1 and call["arguments"]["direction"] == "decline"
    assert len(call["result"]["stores"]) == 1
    assert "SH11 - Chicago, IL" in out["answer"] and "Denver" not in out["answer"] and _store_lines(out["answer"]) == 0
    assert "Five" not in out["answer"]


def test_requested_count_is_respected(ds):
    out = agent.run_rules("Show the top 3 stores with the biggest revenue decline", ds, ds.fact, "full")
    assert out["tool_log"][0]["arguments"]["top_n"] == 3 and _store_lines(out["answer"]) == 3
    plural = agent.run_rules("Which stores had the largest declines?", ds, ds.fact, "full")
    assert _store_lines(plural["answer"]) == 5  # plural wording -> a short list


def test_dollar_vs_percent_basis_follows_the_wording(ds):
    pct = agent.run_rules("Which store had the greatest percentage decline?", ds, ds.fact, "full")["tool_log"][0]
    usd = agent.run_rules("Which store had the greatest revenue decline?", ds, ds.fact, "full")["tool_log"][0]
    assert pct["arguments"]["rank_by"] == "percent" and usd["arguments"]["rank_by"] == "dollars"
    # independent check of the leader on each basis
    s = raw_sales().merge(raw_stores()[["ID", "StoreType"]], left_on="Store ID", right_on="ID")
    s = s[s["StoreType"] != "Website"]
    py = s.pivot_table(index="Store ID", columns=s["Date"].astype(str).str[:4], values="Revenue", aggfunc="sum")
    assert pct["result"]["stores"][0]["store_id"] == ((py["2029"] - py["2025"]) / py["2025"]).idxmin()
    assert usd["result"]["stores"][0]["store_id"] == (py["2029"] - py["2025"]).idxmin()


def test_year_in_the_question_is_used(ds):
    one = agent.run_rules("Which store had the biggest decline in 2026?", ds, ds.fact, "full")["tool_log"][0]
    assert (one["arguments"]["start_year"], one["arguments"]["end_year"]) == (2025, 2026)
    two = agent.run_rules("Which store had the greatest decline from 2026 to 2028?", ds, ds.fact, "full")["tool_log"][0]
    assert (two["arguments"]["start_year"], two["arguments"]["end_year"]) == (2026, 2028)
    assert two["result"]["start_year"] == 2026 and two["result"]["end_year"] == 2028


def test_no_decline_and_no_growth_are_stated_not_invented(ds):
    grown = ds.fact.copy()
    grown["Revenue"] = grown["Revenue"] * (1 + (grown["Year"] - grown["Year"].min()).astype(float))
    ans = agent.run_rules("Which store had the greatest revenue decline?", ds, grown, "full")["answer"]
    assert "No location had revenue decline" in ans
    ans = agent.run_rules("Which store had the greatest revenue growth?", ds, ds.fact, "full")["answer"]
    assert "No location had revenue growth" in ans and "website" in ans.lower()  # physical stores only, website excluded


def test_lowest_and_highest_revenue_store(ds):
    by_store = ds.fact.groupby("Store")["Revenue"].sum()
    low = agent.run_rules("Which store has the lowest revenue?", ds, ds.fact, "full")["tool_log"][0]["result"]["ranking"]
    high = agent.run_rules("Which store has the highest revenue?", ds, ds.fact, "full")["tool_log"][0]["result"]["ranking"]
    assert len(low) == len(high) == 1
    assert low[0]["store"] == by_store.idxmin() and high[0]["store"] == by_store.idxmax()


def test_growth_word_grew_routes_to_growth(ds):
    ans = agent.run_rules("Which category grew the most?", ds, ds.fact, "full")["answer"]
    assert "greatest revenue growth" in ans and "Espresso Machines" in ans


def test_tool_validates_direction_and_rank_by(ds):
    assert "error" in A.get_store_trend(ds.fact, direction="sideways")
    assert "error" in A.get_store_trend(ds.fact, direction="decline", rank_by="vibes")
    r = A.get_store_trend(ds.fact, direction="decline", rank_by="percent", top_n=1)
    assert len(r["stores"]) == 1 and r["stores"][0]["store_id"] == r["largest_pct_decline"]["store_id"]

