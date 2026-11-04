"""
E-Presso BI Analyst - Streamlit application.

Tabs: Dashboard | Data profile & model | Validation | Assistant | About
All numbers are calculated from the data currently loaded (bundled sample or your upload);
nothing is hardcoded. The dashboard and the assistant call the same functions in analytics.py.
"""
from __future__ import annotations

import os
import re
import traceback
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import agent
import analytics as A

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MAX_QUESTIONS = 15
MAX_QUESTION_CHARS = 500

st.set_page_config(page_title="E-Presso BI Analyst", page_icon="☕", layout="wide")


# ------------------------------------------------------------------ helpers
def get_secret(name: str, default=None):
    """Streamlit secrets first, then environment variables. Never printed or logged."""
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001 - no secrets file / missing key
        return os.environ.get(name, default)


def money(x, cents=False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    s = f"${abs(x):,.2f}" if cents else f"${abs(x):,.0f}"
    return f"-{s}" if x < 0 else s


@st.cache_data(show_spinner=False)
def bundled_files():
    keep = {".csv", ".tsv", ".txt", ".xlsx"}
    return tuple((p.name, p.read_bytes()) for p in sorted(DATA_DIR.glob("*")) if p.suffix.lower() in keep)


@st.cache_data(show_spinner="Reading and checking the data...")
def build_from_files(upload_files: tuple, fill_from_bundled: bool):
    """Load uploaded files (or the bundled data), build the model, return (dataset, messages, unrecognised)."""
    if upload_files:
        tables, messages, unrecognised = A.load_tables(list(upload_files))
        label = "Uploaded: " + ", ".join(n for n, _ in upload_files)
        if fill_from_bundled:
            bundled, _, _ = A.load_tables(list(bundled_files()))
            for name, df in bundled.items():
                if name not in tables:
                    tables[name] = df
                    messages.append(f"{name} table was not in your upload, so the bundled {name} table is used.")
                    label += f" (+ bundled {name})"
    else:
        tables, messages, unrecognised = A.load_tables(list(bundled_files()))
        label = "Bundled E-Presso sample data"
    return A.build_dataset(tables, label), messages, unrecognised


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return agent.make_client(api_key)


def data_fingerprint(fact) -> str:
    return f"{len(fact):,} sales rows, total revenue {money(A.total_revenue(fact), cents=True)}"


# ------------------------------------------------------------------ sidebar: data
st.title("☕ E-Presso BI Analyst")
st.caption("Interactive revenue analysis with data checks, validation tools and an analytical assistant.")

with st.sidebar:
    st.header("Data")
    uploads = st.file_uploader(
        "Upload an updated dataset (optional)",
        type=["csv", "tsv", "txt", "xlsx", "xls", "accdb", "mdb"],
        accept_multiple_files=True,
        help="Upload Sales, Stores, Products, Categories and Survey files (CSV or Excel; Access needs loading.py). "
        "Tables are recognised by their columns, not file names. Uploaded files replace the bundled data.",
    )
    fill = st.checkbox("Use bundled data for any table I don't upload", value=True)
    if uploads:
        st.caption("Using your uploaded data. Remove the files to return to the bundled data.")
    else:
        st.caption("Using the bundled E-Presso sample data.")

try:
    ds, load_messages, unrecognised = build_from_files(tuple((f.name, f.getvalue()) for f in uploads), fill)
except ValueError as exc:
    st.error(f"The data could not be prepared: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Unexpected error while loading data: {type(exc).__name__}: {exc}")
    st.code(traceback.format_exc())
    st.stop()

for u in unrecognised:
    st.warning(f"Could not recognise a table in: {u}")

fact = ds.fact
sig = f"{len(fact)}-{fact['Revenue'].sum():.2f}-{fact['Year'].min()}-{fact['Year'].max()}"  # resets filter widgets when data changes

# ------------------------------------------------------------------ sidebar: filters
with st.sidebar:
    st.divider()
    st.header("Filters")
    all_years = sorted(int(y) for y in fact["Year"].dropna().unique())
    all_channels = sorted(fact["Channel"].unique())
    all_cats = sorted(fact["Category"].unique())
    all_stores = sorted(fact["Store"].unique())
    sel_years = st.multiselect("Year", all_years, default=all_years, key=f"yr_{sig}")
    sel_channels = st.multiselect("Sales channel", all_channels, default=all_channels, key=f"ch_{sig}")
    sel_cats = st.multiselect("Product category", all_cats, default=all_cats, key=f"ca_{sig}")
    sel_stores = st.multiselect("Store", all_stores, default=all_stores, key=f"st_{sig}")
    st.caption("Charts, KPIs and the optional filtered assistant scope recalculate when filters change.")

filtered = A.filter_fact(fact, sel_years, sel_channels, sel_cats, sel_stores)
filter_text = A.describe_filters(sel_years, sel_channels, sel_cats, sel_stores, fact)
filters_active = not (
    set(sel_years) == set(all_years) and set(sel_channels) == set(all_channels)
    and set(sel_cats) == set(all_cats) and set(sel_stores) == set(all_stores)
)

tab_dash, tab_profile, tab_validate, tab_assistant, tab_about = st.tabs(
    ["Dashboard", "Data profile & model", "Validation", "Assistant", "About"]
)


# ------------------------------------------------------------------ Dashboard
def render_dashboard():
    st.subheader("Revenue dashboard")
    st.caption(f"**Data:** {ds.source_label}  |  **Filters:** {filter_text}  |  **Period shown:** {A.period_label(filtered)}")
    if filtered.empty:
        st.warning("No records match the selected filters. Widen the filters in the sidebar.")
        return
    f = filtered
    total = A.total_revenue(f)
    by_ch = A.revenue_by(f, "Channel")
    n_months = f.dropna(subset=["Year", "Month"])[["Year", "Month"]].drop_duplicates().shape[0]

    cols = st.columns(2 + min(len(by_ch), 3) + 1)
    cols[0].metric("Total revenue", money(total), help="Sum of Revenue for the filtered rows.")
    for i, row in enumerate(by_ch.head(3).itertuples(index=False), start=1):
        cols[i].metric(f"{row.Channel} revenue", money(row.Revenue), f"{row._2:.1f}% of total", delta_color="off")
    cols[-2].metric("Average per month", money(total / n_months) if n_months else "n/a", help="Total revenue / months with data.")
    cols[-1].metric("Sales rows", f"{len(f):,}")

    period = A.period_label(f)
    left, right = st.columns(2)
    with left:
        by_year = A.revenue_by_year(f, by="Channel")
        fig = px.bar(by_year, x="Year", y="Revenue", color="Channel", barmode="stack",
                     title=f"Revenue by year and channel ({period})", labels={"Revenue": "Revenue ($)"})
        fig.update_xaxes(type="category")
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        yt = A.revenue_by_year(f)
        yt["Change vs prior year"] = yt["Revenue"].pct_change() * 100
        yt["Months of data"] = yt["Year"].map(lambda y: A.months_present(f, y))
        st.dataframe(
            yt.assign(Revenue=yt["Revenue"].map(lambda v: money(v)), **{"Change vs prior year": yt["Change vs prior year"].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}%")}),
            hide_index=True, width="stretch",
        )
    with right:
        m = A.revenue_by_month(f)
        m["Year"] = m["Year"].astype(str)
        fig = px.line(m, x="Month name", y="Revenue", color="Year", markers=True,
                      category_orders={"Month name": A.MONTH_NAMES}, title=f"Monthly revenue pattern ({period})",
                      labels={"Revenue": "Revenue ($)", "Month name": "Month"})
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")
        ms = A.monthly_share(f)
        if not ms.empty:
            piv = ms.pivot(index="Year", columns="Month", values="Share of year %").reindex(columns=range(1, 13))
            piv.columns = A.MONTH_NAMES
            fig = px.imshow(piv, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                            title="Each month as % of its own year's revenue (same filters)",
                            labels={"color": "% of year", "x": "Month", "y": "Year"})
            fig.update_yaxes(type="category")
            st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        n_show = st.slider("Stores to show", 3, max(3, f["Store"].nunique()), min(10, max(3, f["Store"].nunique())), key=f"nshow_{sig}") if f["Store"].nunique() > 3 else f["Store"].nunique()
        by_store = A.revenue_by(f, "Store").head(n_show).iloc[::-1]
        fig = px.bar(by_store, x="Revenue", y="Store", orientation="h", title=f"Revenue by store, top {n_show} ({period})",
                     labels={"Revenue": "Revenue ($)"})
        fig.update_layout(xaxis_tickprefix="$", xaxis_tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")
    with c2:
        by_cat = A.revenue_by(f, "Category")
        fig = px.bar(by_cat, x="Category", y="Revenue", title=f"Revenue by product category ({period})",
                     labels={"Revenue": "Revenue ($)"}, text=by_cat["Share %"].map(lambda v: f"{v:.1f}%"))
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Filtered records"):
        st.dataframe(f.head(500), hide_index=True, width="stretch")
        st.download_button("Download filtered records (CSV)", f.to_csv(index=False).encode(), "epresso_filtered.csv", "text/csv")


# ------------------------------------------------------------------ Profile & model
def render_profile():
    st.subheader("Data profile and model")
    p = A.profile_dataset(ds)
    for msg in load_messages:
        st.caption("• " + msg)
    st.markdown("#### Tables loaded")
    st.dataframe(p["summary"], hide_index=True, width="stretch")
    st.markdown("#### Grain statements (what one row means)")
    st.dataframe(p["grain"], hide_index=True, width="stretch")
    st.markdown("#### Table relationships (all many-to-one)")
    st.dataframe(p["relationships"], hide_index=True, width="stretch")
    st.caption(
        "The model joins Sales -> Stores (Store ID), Products (Product ID) and Categories (Category ID). "
        "Survey is store-by-year, so it is **never joined row-by-row to Sales** (that would repeat each survey row for "
        "every sale); it is only matched on Store ID + Year after revenue is totalled."
    )
    st.markdown("#### Join reconciliation (row count and revenue must not change)")
    rec = p["reconciliation"].copy()
    rec["Total revenue"] = rec["Total revenue"].map(lambda v: money(v, cents=True))
    st.dataframe(rec, hide_index=True, width="stretch")
    if p["reconciliation"]["OK"].all():
        st.success("Every join preserved the row count and the revenue total.")
    st.markdown("#### Missing values")
    if p["missing"].empty:
        st.success("No missing values in any table.")
    else:
        st.dataframe(p["missing"], hide_index=True, width="stretch")
    st.markdown("#### Other data-quality checks")
    st.dataframe(pd.DataFrame({"Check": list(p["checks"]), "Count": list(p["checks"].values())}), hide_index=True, width="stretch")
    if p["notes"]:
        st.markdown("#### Warnings")
        for n in p["notes"]:
            st.warning(n)
    st.markdown("#### Preview a table")
    name = st.selectbox("Table", list(ds.tables), key=f"prev_{sig}")
    st.dataframe(ds.tables[name].head(100), hide_index=True, width="stretch")
    st.markdown("**Revenue definition:** Revenue is a stored dollar amount per store x product x month in the Sales table "
                "(it is not calculated from quantity x price). Total revenue = sum of that column.")


# ------------------------------------------------------------------ Validation
def validation_rows() -> pd.DataFrame:
    full = ds.fact
    rows = [("Total revenue", A.total_revenue(full))]
    for r in A.revenue_by(full, "Channel").sort_values("Channel").itertuples(index=False):
        rows.append((f"{r.Channel} revenue", r.Revenue))
    for r in A.revenue_by_year(full).itertuples(index=False):
        rows.append((f"Annual revenue {r.Year}", r.Revenue))
    mm = full.dropna(subset=["Month"]).groupby("Month")["Revenue"].sum()
    for mth, v in mm.items():
        rows.append((f"Monthly revenue - {A.MONTH_NAMES[int(mth) - 1]} (all years)", v))
    for r in A.revenue_by(full, "Category").sort_values("Category").itertuples(index=False):
        rows.append((f"Additional: {r.Category} revenue", r.Revenue))
    return pd.DataFrame(rows, columns=["Metric", "Streamlit result"])


def snapshot() -> dict:
    full = ds.fact
    return {
        "label": ds.source_label, "fingerprint": data_fingerprint(full), "total": A.total_revenue(full),
        "by_year": A.revenue_by_year(full).set_index("Year")["Revenue"],
        "by_channel": full.groupby("Channel")["Revenue"].sum(),
        "by_month": full.dropna(subset=["Year", "Month"]).groupby(["Year", "Month"])["Revenue"].sum(),
    }


def render_validation():
    st.subheader("Validation against Power BI")
    st.info("Validation always uses the **full, unfiltered** data, so it is comparable to the Power BI report totals.")
    base = validation_rows()
    base["Power BI result"] = ""
    st.markdown("**Step 1.** Type your Power BI value for each metric (numbers only; $ and commas are fine). "
                "Streamlit results are calculated live from the loaded data.")
    edited = st.data_editor(
        base[["Metric", "Power BI result", "Streamlit result"]], hide_index=True, width="stretch",
        disabled=["Metric", "Streamlit result"], key=f"val_{sig}",
        column_config={"Streamlit result": st.column_config.NumberColumn(format="$%.2f")},
    )
    pb = pd.to_numeric(edited["Power BI result"].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    out = edited.copy()
    out["Power BI result"] = pb
    out["Difference"] = out["Streamlit result"] - pb
    out["Match?"] = ["" if pd.isna(d) else ("Yes" if abs(d) < 0.5 else "NO - investigate") for d in out["Difference"]]
    st.markdown("**Step 2.** Results table (copy into your report; investigate every 'NO').")
    st.dataframe(out, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="$%.2f") for c in ["Power BI result", "Streamlit result", "Difference"]})
    st.download_button("Download validation table (CSV)", out.to_csv(index=False).encode(), "validation_table.csv", "text/csv")
    st.caption("If a value differs: check Power BI filters/relationships (e.g. a Sales -> Survey relationship used for filtering), "
               "blank-key rows, and rounding. Power BI is not automatically correct.")

    st.divider()
    st.subheader("Modified-data ($1,000) test helper")
    st.markdown(
        "1. With the **original** data loaded, click *Save baseline*.\n"
        "2. Create the modified copy (`python scripts/make_modified_data.py`) and upload its files in the sidebar.\n"
        "3. Come back here: the total should be **exactly $1,000** higher, and exactly one year, one month and one channel should change.\n"
        "4. Remove the uploaded files to restore the original data."
    )
    if st.button("Save baseline from the currently loaded data", key="save_base"):
        st.session_state["baseline"] = snapshot()
    b = st.session_state.get("baseline")
    if b:
        cur = snapshot()
        st.markdown(f"**Baseline:** {b['label']} - {b['fingerprint']}")
        st.markdown(f"**Current:** {cur['label']} - {cur['fingerprint']}")
        diff = cur["total"] - b["total"]
        if abs(diff) < 0.005:
            st.info("Current total equals the baseline (no change loaded yet).")
        elif abs(diff - 1000) < 0.005:
            st.success(f"PASS: total revenue increased by exactly {money(diff, cents=True)}.")
        else:
            st.error(f"Total changed by {money(diff, cents=True)}, which is not exactly $1,000.00.")
        d_year = (cur["by_year"] - b["by_year"].reindex(cur["by_year"].index, fill_value=0)).rename("Change")
        d_ch = (cur["by_channel"] - b["by_channel"].reindex(cur["by_channel"].index, fill_value=0)).rename("Change")
        d_mo = (cur["by_month"] - b["by_month"].reindex(cur["by_month"].index, fill_value=0)).rename("Change")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Years that changed**")
        c1.dataframe(d_year[d_year.abs() > 0.005], width="stretch")
        c2.markdown("**Channels that changed**")
        c2.dataframe(d_ch[d_ch.abs() > 0.005], width="stretch")
        c3.markdown("**Year-months that changed**")
        c3.dataframe(d_mo[d_mo.abs() > 0.005], width="stretch")

    st.divider()
    st.subheader("Independent verification")
    st.markdown("Run `python scripts/independent_check.py` (reads the raw CSVs without using this app's code) and compare its "
                "total, channel, annual and monthly figures with the table above. Save the output as evidence.")


# ------------------------------------------------------------------ Assistant
def render_assistant():
    st.subheader("Analytical assistant")
    api_key = get_secret("GEMINI_API_KEY")
    model = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
    max_q = int(get_secret("MAX_QUESTIONS", DEFAULT_MAX_QUESTIONS))
    force_rules = st.checkbox("Use the rules-based assistant only (no language model)", value=False, key="force_rules")
    client = None
    if api_key and not force_rules:
        try:
            client = get_client(str(api_key))
        except agent.LLMUnavailable as exc:
            st.warning(f"Language model unavailable ({exc}); using the rules-based assistant.")
    if client:
        st.success(f"Mode: language-model agent ({model}, Google Gemini function calling) with automatic fallback to the rules-based assistant.")
    else:
        st.info("Mode: **rules-based assistant** (keyword routing to real analytical functions; not a language model).")
    st.caption("The assistant can only call approved functions: " + ", ".join(agent.TOOLS) +
               ". It cannot run code or change data. Every answer shows a tool-call log.")

    scope_choice = st.radio(
        "Answer from", ["Full dataset (ignore dashboard filters)", "Dashboard-filtered subset"], horizontal=True, key="scope_choice"
    )
    if scope_choice.startswith("Full"):
        scope, data_for_agent = f"FULL dataset (dashboard filters not applied) - {data_fingerprint(fact)}", fact
    else:
        scope = f"DASHBOARD-FILTERED subset ({filter_text}) - {data_fingerprint(filtered)}" if filters_active else \
                f"Dashboard-filtered subset (no filters currently active, so same as full dataset) - {data_fingerprint(filtered)}"
        data_for_agent = filtered
    st.caption(f"Current scope: {scope}")

    st.session_state.setdefault("n_questions", 0)
    st.session_state.setdefault("history", [])
    left = max_q - st.session_state["n_questions"]
    st.caption(f"Questions left this session: **{max(left, 0)}** of {max_q}")

    examples = [
        "How much revenue did the website generate in 2028?",
        "Which store experienced the greatest percentage revenue decline?",
        "Which sales channel experienced the greatest revenue growth?",
        "Which E-Presso store is the most profitable?",
        "Does higher customer satisfaction cause higher revenue?",
        "Should E-Presso close its Chicago store?",
        "Which month is busiest?",
    ]
    with st.expander("Example questions (click to ask)"):
        for i, q in enumerate(examples):
            if st.button(q, key=f"ex_{i}"):
                st.session_state["pending_q"] = q

    for item in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            ans = item["answer"]
            st.caption(f"Mode: {ans['mode']}  |  Scope: {ans['scope']}")
            for note in ans.get("notes", []):
                st.warning(note)
            st.markdown(ans["answer"])
            with st.expander(f"Tool-call log ({len(ans['tool_log'])} call(s))"):
                if not ans["tool_log"]:
                    st.write("No function was called for this question.")
                for call in ans["tool_log"]:
                    st.markdown(f"**Function:** `{call['function']}`  \n**Arguments:** `{call['arguments']}`  \n**Scope:** {call['scope']}")
                    st.json(call["result"], expanded=False)

    question = st.chat_input("Ask a business question about the loaded data", max_chars=MAX_QUESTION_CHARS)
    question = question or st.session_state.pop("pending_q", None)
    if question:
        if st.session_state["n_questions"] >= max_q:
            st.warning("Question limit reached for this session. Reload the page to start a new session.")
        elif data_for_agent.empty:
            st.warning("The current scope has no rows. Widen the dashboard filters or choose the full dataset.")
        else:
            st.session_state["n_questions"] += 1
            with st.spinner("Calculating..."):
                ans = agent.answer_question(question, ds, data_for_agent, scope, client=client, model=model)
            st.session_state["history"].append({"question": question, "answer": ans})
            st.rerun()
    if st.session_state["history"] and st.button("Clear conversation (does not reset the question limit)"):
        st.session_state["history"] = []
        st.rerun()


# ------------------------------------------------------------------ About
def render_about():
    st.subheader("About this application")
    st.markdown(
        """
**Purpose.** Explore E-Presso revenue by year, month, channel, store and category, check the data, and ask an analytical assistant.

**Definitions**
- *Revenue* = the stored dollar amount per store x product x month in the Sales table. It is **not profit**; no cost data exists.
- *Channel* = the store type: `Website` or `Brick & Mortar`.
- *Monthly share* = a month's revenue / that calendar year's revenue (same filters).
- *Percent change* = (end - start) / start; undefined when the start value is zero.

**Data model.** Sales (one row per store x product x month) joins many-to-one to Stores, Products and Categories. Survey (one row per store x year)
is kept separate. Every join is checked to keep the row count and revenue total unchanged. Products are joined by ID because product names repeat across categories.

**The assistant.** Questions are routed to approved functions in `analytics.py` through a whitelisted dispatcher. With a Gemini API key configured it is a
language-model agent (function calling); without one (or if the API fails or the quota runs out) it is a rules-based assistant. It never executes generated code,
never edits data, and every answer lists the function, arguments and raw result.

**Known limitations**
- Revenue only: no costs, units, prices, or customer-level data. Satisfaction survey has no response counts or scale definition and the website has no survey scores.
- Comparisons that involve a year with fewer than 12 months are flagged as not like-for-like.
- Correlation shown by the assistant is not causation.
        """
    )


# ------------------------------------------------------------------ render (errors in one tab never hide the others)
for tab, fn in [(tab_dash, render_dashboard), (tab_profile, render_profile), (tab_validate, render_validation),
                (tab_assistant, render_assistant), (tab_about, render_about)]:
    with tab:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - show the real error (Streamlit Cloud redacts it otherwise)
            st.error(f"{fn.__name__} failed: {type(exc).__name__}: {exc}")
            st.code(traceback.format_exc())
