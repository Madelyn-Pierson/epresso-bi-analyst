"""
analytics.py - E-Presso data loading, preparation, profiling and analytical functions.

Everything the dashboard and the assistant report is calculated here, from whatever
data is currently loaded. Nothing is hardcoded: years, stores, channels and categories
are all discovered from the data, so an updated dataset works without code changes.

Layout
  1. Loading      - read CSV/Excel files, recognise which table each one is
  2. Preparation  - clean types, join lookup tables (many-to-one only), reconcile totals
  3. Profiling    - row counts, missing values, duplicates, orphan keys, grain checks
  4. Core metrics - small DataFrame helpers shared by the dashboard AND the assistant
  5. Tools        - the analytical functions the assistant is allowed to call
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1. LOADING
# --------------------------------------------------------------------------------------

# canonical column name -> accepted normalised header spellings
TABLE_SPECS: dict[str, dict[str, list[str]]] = {
    "sales": {
        "Sales ID": ["salesid"],
        "Date": ["date"],
        "Store ID": ["storeid"],
        "Category ID": ["categoryid"],
        "Product ID": ["productid"],
        "Revenue": ["revenue"],
    },
    "survey": {
        "Survey ID": ["surveyid"],
        "Store ID": ["storeid"],
        "Year": ["year"],
        "Satisfaction": ["satisfaction"],
        "Interest": ["interest"],
    },
    "stores": {
        "ID": ["id", "storeid"],
        "City": ["city"],
        "State": ["state"],
        "StoreType": ["storetype"],
    },
    "products": {"Product ID": ["productid"], "Product": ["product", "productname"]},
    "categories": {"Category ID": ["categoryid"], "Category": ["category", "categoryname"]},
}
CHECK_ORDER = ["sales", "survey", "stores", "products", "categories"]
REQUIRED_TABLES = ["sales", "stores"]
KEYS = {
    "sales": ["Sales ID"],
    "products": ["Product ID"],
    "categories": ["Category ID"],
    "stores": ["ID"],
    "survey": ["Survey ID"],
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _classify(df: pd.DataFrame):
    """Return (table_name, rename_map) if df has the columns of a known table, else None."""
    normed = {_norm(c): c for c in df.columns}
    for table in CHECK_ORDER:
        rename, ok = {}, True
        for canon, variants in TABLE_SPECS[table].items():
            hit = next((normed[v] for v in variants if v in normed), None)
            if hit is None:
                ok = False
                break
            rename[hit] = canon
        if ok:
            return table, rename
    return None


def _find_header(raw: pd.DataFrame):
    """Find the header row (some exports have a title row above the headers)."""
    for i in range(min(6, len(raw))):
        cand = raw.iloc[i + 1 :].copy()
        cand.columns = [str(c).strip() if pd.notna(c) else f"col{j}" for j, c in enumerate(raw.iloc[i])]
        found = _classify(cand)
        if found:
            return cand.reset_index(drop=True), found
    return None, None


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    try:
        delim = csv.Sniffer().sniff(text[:4096], delimiters=",\t;|").delimiter
    except csv.Error:
        delim = ","
    return pd.read_csv(io.StringIO(text), sep=delim, header=None, dtype=str, keep_default_na=True)


def _read_access(data: bytes) -> dict[str, pd.DataFrame]:
    """Read every user table in an Access (.accdb/.mdb) file with the pure-Python access-parser package."""
    import tempfile

    from access_parser import AccessParser

    with tempfile.NamedTemporaryFile(suffix=".accdb") as tmp:
        tmp.write(data)
        tmp.flush()
        db = AccessParser(tmp.name)
        out = {}
        for name in db.catalog:
            if name.startswith("MSys") or name.startswith("f_"):  # Access system / attachment tables
                continue
            parsed = db.parse_table(name)
            if parsed:
                out[name] = pd.DataFrame(parsed)
    return out


def _read_other(fname: str, data: bytes) -> dict[str, pd.DataFrame]:
    """
    Access files are read natively. Other database formats (e.g. SQLite) are delegated to the
    starter template's loading.py if it is present. Returns {table label: DataFrame with real headers}.
    """
    if fname.lower().endswith((".accdb", ".mdb")):
        try:
            return _read_access(data)
        except ImportError:
            pass  # fall through to the starter's loader
    try:
        import loading  # starter-template module
    except ImportError as exc:
        raise ValueError(
            "this file type needs the access-parser package (Access) or loading.py from the course starter. "
            "Alternatively export the tables to CSV or Excel."
        ) from exc
    res = loading.load_file(data, fname)
    return {t.label: t.df for t in res.tables if getattr(t, "usable", True)}


def load_tables(files: list[tuple[str, bytes]]):
    """
    Recognise E-Presso tables in a list of (filename, bytes).
    CSV/TSV/TXT, Excel and Access are read natively (a workbook may hold several sheets, a title row above
    the headers is tolerated, and sheets that are not data tables - e.g. documentation - are ignored).
    Returns (tables, messages, unrecognised).
    """
    tables: dict[str, pd.DataFrame] = {}
    messages: list[str] = []
    unrecognised: list[str] = []
    for fname, data in files:
        lower = fname.lower()
        try:
            if lower.endswith((".xlsx", ".xlsm", ".xls")):
                sheets = {k: (v, True) for k, v in pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=object).items()}
            elif lower.endswith((".csv", ".tsv", ".txt")):
                sheets = {"": (_read_csv_bytes(data), True)}
            else:
                sheets = {k: (v, False) for k, v in _read_other(fname, data).items()}
        except Exception as exc:  # noqa: BLE001 - report any reader failure to the user
            unrecognised.append(f"{fname} (could not be read: {exc})")
            continue
        file_hits, skipped = 0, []
        for sheet, (raw, headerless) in sheets.items():
            label = f"{fname}" + (f" [{sheet}]" if sheet else "")
            raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
            if raw.empty:
                continue
            if headerless:
                df, found = _find_header(raw)
            else:
                found, df = _classify(raw), raw
            if not found:
                skipped.append(label)
                continue
            file_hits += 1
            table, rename = found
            df = df.rename(columns=rename)
            if table in tables:
                messages.append(f"'{label}' looks like the {table} table, but {table} was already loaded - ignored.")
                continue
            tables[table] = df
            messages.append(f"{label} -> {table} table ({len(df):,} rows)")
        if file_hits:  # other sheets/tables in a file that did contain data are just ignored
            messages += [f"{s} is not an E-Presso data table - ignored." for s in skipped]
        else:
            unrecognised += skipped or [fname]
    return tables, messages, unrecognised


# --------------------------------------------------------------------------------------
# 2. PREPARATION
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    tables: dict[str, pd.DataFrame]  # cleaned source tables (canonical column names)
    fact: pd.DataFrame  # sales joined to stores / products / categories (one row per sales row)
    survey: pd.DataFrame  # survey results, one row per store-year
    notes: list[str] = field(default_factory=list)  # warnings found while preparing
    reconciliation: list[dict] = field(default_factory=list)  # row/revenue checks after each join
    source_label: str = ""


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def _year_from(series: pd.Series) -> pd.Series:
    """'Y2025' / '2025' / 2025 -> 2025 (Int64)."""
    digits = series.astype(str).str.extract(r"(\d{4})")[0]
    return pd.to_numeric(digits, errors="coerce").astype("Int64")


def _dedupe_lookup(df: pd.DataFrame, key: str, name: str, notes: list[str]) -> pd.DataFrame:
    dups = int(df[key].duplicated().sum())
    if dups:
        notes.append(
            f"{name}: {dups} duplicate value(s) of key '{key}'. Kept the first row of each so joins cannot duplicate revenue."
        )
        df = df.drop_duplicates(subset=key, keep="first")
    return df


def build_dataset(tables: dict[str, pd.DataFrame], source_label: str = "") -> Dataset:
    """Clean the tables and join the lookups onto Sales. Raises ValueError if required tables are missing."""
    missing = [t for t in REQUIRED_TABLES if t not in tables]
    if missing:
        raise ValueError(
            "Required table(s) not found: " + ", ".join(missing) + ". Upload at least the Sales and Stores tables."
        )
    notes: list[str] = []
    recon: list[dict] = []

    # --- Sales (grain: one row per store x product x month) ---
    sales = tables["sales"].copy()
    for col in ["Sales ID", "Store ID", "Category ID", "Product ID"]:
        sales[col] = sales[col].astype("string").str.strip()
    raw_rev = sales["Revenue"]
    sales["Revenue"] = _to_number(raw_rev)
    bad_rev = int((raw_rev.notna() & sales["Revenue"].isna()).sum())
    null_rev = int(raw_rev.isna().sum())
    if bad_rev:
        notes.append(f"Sales: {bad_rev} Revenue value(s) are not numeric and were treated as missing.")
    if null_rev:
        notes.append(f"Sales: {null_rev} row(s) have no Revenue value.")
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    bad_dates = int(sales["Date"].isna().sum())
    if bad_dates:
        lost = float(sales.loc[sales["Date"].isna(), "Revenue"].sum())
        notes.append(
            f"Sales: {bad_dates} row(s) have a missing/invalid Date (revenue {lost:,.2f}). "
            "They count toward total revenue but cannot appear in year/month views."
        )
    sales["Year"] = sales["Date"].dt.year.astype("Int64")
    sales["Month"] = sales["Date"].dt.month.astype("Int64")
    dup_ids = int(sales["Sales ID"].duplicated().sum())
    if dup_ids:
        notes.append(f"Sales: {dup_ids} duplicate Sales ID value(s). Rows were NOT removed - please review.")
    neg = int((sales["Revenue"] < 0).sum())
    if neg:
        notes.append(f"Sales: {neg} row(s) have negative revenue (returns/credits?).")

    base_rows, base_rev = len(sales), float(sales["Revenue"].sum())
    recon.append({"Step": "Sales as loaded", "Rows": base_rows, "Total revenue": base_rev, "OK": True})
    fact = sales

    def _check(step: str, df: pd.DataFrame):
        ok = len(df) == base_rows and math.isclose(float(df["Revenue"].sum()), base_rev, abs_tol=0.005)
        recon.append({"Step": step, "Rows": len(df), "Total revenue": float(df["Revenue"].sum()), "OK": ok})
        if not ok:
            raise ValueError(f"Join '{step}' changed row count or revenue - the relationship is not many-to-one.")

    # --- Stores (many-to-one on Store ID) ---
    stores = tables["stores"].copy()
    stores["ID"] = stores["ID"].astype("string").str.strip()
    stores = _dedupe_lookup(stores, "ID", "Stores", notes)
    for col in ["Market", "Location", "PostalCode"]:
        if col not in stores.columns:
            stores[col] = pd.NA
    st_cols = stores[["ID", "City", "State", "StoreType", "Market", "Location"]].rename(
        columns={"ID": "Store ID", "StoreType": "Channel"}
    )
    fact = fact.merge(st_cols, on="Store ID", how="left", validate="m:1")
    _check("+ Stores (Store ID -> ID)", fact)
    fact["Channel"] = fact["Channel"].fillna("Unknown")
    has_city = fact["City"].notna() & (fact["City"].astype(str).str.strip() != "")
    fact["Store"] = np.where(
        has_city,
        fact["Store ID"].astype(str) + " - " + fact["City"].astype(str) + ", " + fact["State"].astype(str),
        fact["Store ID"].astype(str) + " - " + fact["Channel"].astype(str),
    )

    # --- Categories ---
    if "categories" in tables:
        cats = tables["categories"].copy()
        cats["Category ID"] = cats["Category ID"].astype("string").str.strip()
        cats = _dedupe_lookup(cats, "Category ID", "Categories", notes)
        fact = fact.merge(cats[["Category ID", "Category"]], on="Category ID", how="left", validate="m:1")
        _check("+ Categories (Category ID)", fact)
    else:
        notes.append("Categories table not loaded - categories shown by ID.")
        fact["Category"] = np.nan
    fact["Category"] = fact["Category"].fillna(fact["Category ID"]).fillna("Unknown")

    # --- Products (names are NOT unique across categories, so always join on Product ID) ---
    if "products" in tables:
        prods = tables["products"].copy()
        prods["Product ID"] = prods["Product ID"].astype("string").str.strip()
        prods = _dedupe_lookup(prods, "Product ID", "Products", notes)
        dup_names = int(prods["Product"].duplicated().sum())
        if dup_names:
            notes.append(
                f"Products: {dup_names} product name(s) repeat under different Product IDs. "
                "Joins use Product ID, and charts label products as 'Name (Category)'."
            )
        fact = fact.merge(prods[["Product ID", "Product"]], on="Product ID", how="left", validate="m:1")
        _check("+ Products (Product ID)", fact)
    else:
        notes.append("Products table not loaded - products shown by ID.")
        fact["Product"] = np.nan
    fact["Product"] = fact["Product"].fillna(fact["Product ID"]).fillna("Unknown")
    fact["Product label"] = fact["Product"].astype(str) + " (" + fact["Category"].astype(str) + ")"

    fact = fact[
        [
            "Sales ID", "Date", "Year", "Month", "Store ID", "Store", "City", "State", "Market", "Location",
            "Channel", "Category ID", "Category", "Product ID", "Product", "Product label", "Revenue",
        ]
    ].reset_index(drop=True)

    # --- Survey (grain: one row per store x year; kept separate so it can never duplicate revenue) ---
    if "survey" in tables:
        sv = tables["survey"].copy()
        sv["Store ID"] = sv["Store ID"].astype("string").str.strip()
        sv["Year"] = _year_from(sv["Year"])
        sv["Satisfaction"] = pd.to_numeric(sv["Satisfaction"], errors="coerce")
        sv["Interest"] = pd.to_numeric(sv["Interest"], errors="coerce")
        dups = int(sv.duplicated(subset=["Store ID", "Year"]).sum())
        if dups:
            notes.append(f"Survey: {dups} duplicate Store ID + Year row(s). Kept the first of each.")
            sv = sv.drop_duplicates(subset=["Store ID", "Year"], keep="first")
        survey = sv[["Store ID", "Year", "Satisfaction", "Interest"]].reset_index(drop=True)
        miss = int(survey["Satisfaction"].isna().sum())
        if miss:
            notes.append(
                f"Survey: {miss} store-year row(s) have no Satisfaction score "
                "(e.g. the website has no survey), which limits satisfaction conclusions."
            )
    else:
        survey = pd.DataFrame(columns=["Store ID", "Year", "Satisfaction", "Interest"])
        notes.append("Survey table not loaded - satisfaction analysis unavailable.")

    return Dataset(tables=tables, fact=fact, survey=survey, notes=notes, reconciliation=recon, source_label=source_label)


# --------------------------------------------------------------------------------------
# 3. PROFILING
# --------------------------------------------------------------------------------------


def profile_dataset(ds: Dataset) -> dict:
    """Everything the 'Data profile' page shows. All values are computed from the loaded tables."""
    summary, missing_rows = [], []
    for name in CHECK_ORDER:
        if name not in ds.tables:
            continue
        df = ds.tables[name]
        key = KEYS[name]
        summary.append(
            {
                "Table": name,
                "Rows": len(df),
                "Columns": df.shape[1],
                "Missing cells": int(df.isna().sum().sum()),
                "Duplicate rows": int(df.duplicated().sum()),
                f"Duplicate key values": int(df[key[0]].duplicated().sum()),
            }
        )
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                missing_rows.append({"Table": name, "Column": col, "Missing": n, "% of rows": round(100 * n / len(df), 1)})

    t = ds.tables
    rels = []

    def _rel(child, ccol, parent, pcol):
        if child in t and parent in t:
            cvals = set(t[child][ccol].dropna().astype(str).str.strip())
            pvals = set(t[parent][pcol].dropna().astype(str).str.strip())
            rels.append(
                {
                    "Child": f"{child}.{ccol}",
                    "Parent": f"{parent}.{pcol}",
                    "Declared": "many-to-one",
                    "Parent key unique?": bool(t[parent][pcol].dropna().is_unique),
                    "Orphan values (child with no parent)": len(cvals - pvals),
                }
            )

    _rel("sales", "Store ID", "stores", "ID")
    _rel("sales", "Product ID", "products", "Product ID")
    _rel("sales", "Category ID", "categories", "Category ID")
    _rel("survey", "Store ID", "stores", "ID")
    if "survey" in t:
        sales_sy = ds.fact.dropna(subset=["Year"])[["Store ID", "Year"]].drop_duplicates()
        merged = sales_sy.merge(ds.survey[["Store ID", "Year"]], how="left", on=["Store ID", "Year"], indicator=True)
        rels.append(
            {
                "Child": "sales (Store ID + Year)",
                "Parent": "survey (Store ID + Year)",
                "Declared": "many-to-one",
                "Parent key unique?": bool(not ds.survey.duplicated(subset=["Store ID", "Year"]).any()),
                "Orphan values (child with no parent)": int((merged["_merge"] == "left_only").sum()),
            }
        )

    f = ds.fact
    grain_dups = int(f.duplicated(subset=["Store ID", "Product ID", "Year", "Month"]).sum())
    cat_per_prod = int((f.groupby("Product ID")["Category ID"].nunique() > 1).sum())
    months_per_sy = f.dropna(subset=["Year"]).groupby(["Store ID", "Year"])["Month"].nunique()
    incomplete = int((months_per_sy < 12).sum())

    prod_names_dup = 0
    if "products" in t:
        prod_names_dup = int(t["products"]["Product"].duplicated().sum())

    grain = [
        {
            "Table": "sales",
            "Grain (one row = ...)": "one product's revenue at one location (store or website) in one month",
            "Checked in the data": f"{grain_dups} duplicate Store+Product+Month combinations; "
            f"{cat_per_prod} products appear under more than one category",
        },
        {
            "Table": "stores",
            "Grain (one row = ...)": "one sales location (a physical store or the website)",
            "Checked in the data": f"{len(ds.tables['stores']):,} locations; key 'ID' unique: "
            f"{ds.tables['stores']['ID'].is_unique}",
        },
    ]
    if "products" in t:
        grain.append(
            {
                "Table": "products",
                "Grain (one row = ...)": "one product (identified by Product ID, not by name)",
                "Checked in the data": f"{len(t['products']):,} products; {prod_names_dup} repeated product names",
            }
        )
    if "categories" in t:
        grain.append(
            {
                "Table": "categories",
                "Grain (one row = ...)": "one product category",
                "Checked in the data": f"{len(t['categories']):,} categories",
            }
        )
    if "survey" in t:
        grain.append(
            {
                "Table": "survey",
                "Grain (one row = ...)": "one location's survey result for one year (NOT per sale)",
                "Checked in the data": f"{len(ds.survey):,} location-years; duplicate Store+Year: "
                f"{int(ds.survey.duplicated(subset=['Store ID', 'Year']).sum())}",
            }
        )

    checks = {
        "Duplicate Sales IDs": int(ds.fact["Sales ID"].duplicated().sum()),
        "Rows with missing Revenue": int(ds.fact["Revenue"].isna().sum()),
        "Rows with negative Revenue": int((ds.fact["Revenue"] < 0).sum()),
        "Rows with zero Revenue": int((ds.fact["Revenue"] == 0).sum()),
        "Rows with missing/invalid Date": int(ds.fact["Date"].isna().sum()),
        "Store-years with fewer than 12 months of sales": incomplete,
        "Unknown channel rows (store not in Stores table)": int((ds.fact["Channel"] == "Unknown").sum()),
    }
    return {
        "summary": pd.DataFrame(summary),
        "missing": pd.DataFrame(missing_rows, columns=["Table", "Column", "Missing", "% of rows"]),
        "relationships": pd.DataFrame(rels),
        "grain": pd.DataFrame(grain),
        "checks": checks,
        "reconciliation": pd.DataFrame(ds.reconciliation),
        "notes": ds.notes,
    }


# --------------------------------------------------------------------------------------
# 4. CORE METRICS (shared by the dashboard and the assistant, so they always agree)
# --------------------------------------------------------------------------------------


def filter_fact(fact, years=None, channels=None, categories=None, stores=None) -> pd.DataFrame:
    """Dashboard filters. None means 'no filter'; an empty list means 'nothing selected' (no rows)."""
    mask = pd.Series(True, index=fact.index)
    if years is not None:
        mask &= fact["Year"].isin(list(years))
    if channels is not None:
        mask &= fact["Channel"].isin(list(channels))
    if categories is not None:
        mask &= fact["Category"].isin(list(categories))
    if stores is not None:
        mask &= fact["Store"].isin(list(stores))
    return fact[mask]


def total_revenue(fact) -> float:
    return float(fact["Revenue"].sum())


def revenue_by_year(fact, by: str | None = None) -> pd.DataFrame:
    d = fact.dropna(subset=["Year"])
    cols = ["Year"] + ([by] if by else [])
    out = d.groupby(cols, as_index=False)["Revenue"].sum()
    out["Year"] = out["Year"].astype(int)
    return out.sort_values(cols).reset_index(drop=True)


def revenue_by_month(fact) -> pd.DataFrame:
    d = fact.dropna(subset=["Year", "Month"])
    out = d.groupby(["Year", "Month"], as_index=False)["Revenue"].sum()
    out["Year"] = out["Year"].astype(int)
    out["Month"] = out["Month"].astype(int)
    out["Month name"] = out["Month"].map(lambda m: MONTH_NAMES[m - 1])
    return out.sort_values(["Year", "Month"]).reset_index(drop=True)


def monthly_share(fact) -> pd.DataFrame:
    """Monthly revenue as % of the SAME calendar year's revenue (same filters)."""
    m = revenue_by_month(fact)
    year_total = m.groupby("Year")["Revenue"].transform("sum")
    m["Share of year %"] = np.where(year_total != 0, 100 * m["Revenue"] / year_total, np.nan)
    return m


def revenue_by(fact, dim: str) -> pd.DataFrame:
    out = fact.groupby(dim, as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    tot = out["Revenue"].sum()
    out["Share %"] = np.where(tot != 0, 100 * out["Revenue"] / tot, np.nan)
    return out.reset_index(drop=True)


def period_label(fact) -> str:
    d = fact["Date"].dropna()
    if d.empty:
        return "no dated rows"
    return f"{d.min():%b %Y} - {d.max():%b %Y}"


def months_present(fact, year: int) -> int:
    return int(fact.loc[fact["Year"] == year, "Month"].nunique())


def describe_filters(years, channels, categories, stores, fact_all) -> str:
    """Human-readable description of which filters are active (for chart titles / labels)."""
    parts = []

    def _desc(label, chosen, universe):
        if chosen is None or set(chosen) == set(universe):
            return f"all {label}"
        return f"{label}: " + ", ".join(map(str, chosen)) if chosen else f"no {label}"

    parts.append(_desc("years", sorted(years) if years is not None else None, sorted(fact_all["Year"].dropna().unique())))
    parts.append(_desc("channels", channels, fact_all["Channel"].unique()))
    parts.append(_desc("categories", categories, fact_all["Category"].unique()))
    parts.append(_desc("stores", stores, fact_all["Store"].unique()))
    return "; ".join(parts)


# --------------------------------------------------------------------------------------
# 5. TOOLS - the only functions the assistant may call
# --------------------------------------------------------------------------------------
# Each tool: takes the data (and simple, validated arguments), returns a JSON-safe dict.
# They never modify data and never execute model-written code.

_SYNONYMS = {
    "online": "website", "web": "website", "ecommerce": "website", "e-commerce": "website",
    "physical": "brick & mortar", "retail": "brick & mortar", "in-store": "brick & mortar",
    "brick and mortar": "brick & mortar", "brick-and-mortar": "brick & mortar", "stores": "brick & mortar",
    "store": "brick & mortar",
}


def _clean(obj):
    """Make results JSON-safe: numpy -> python, NaN -> None, Timestamp -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.to_dict(orient="records"))


def _match_option(value, options, what: str):
    """Case-insensitive match of a user value to the options in the data. Returns (match, error)."""
    options = [str(o) for o in options]
    low = str(value).strip().lower()
    low = _SYNONYMS.get(low, low)
    exact = [o for o in options if o.lower() == low]
    if exact:
        return exact[0], None
    part = [o for o in options if low and low in o.lower()]
    if len(part) == 1:
        return part[0], None
    return None, {"error": f"Unknown {what} '{value}'.", "valid_values": sorted(options)}


def _match_store(fact, value):
    """Match by Store ID, city or label. Returns (list_of_store_ids, error)."""
    low = str(value).strip().lower()
    stores = fact[["Store ID", "Store", "City"]].drop_duplicates()
    ids = stores[stores["Store ID"].astype(str).str.lower() == low]["Store ID"].tolist()
    if not ids:
        ids = stores[stores["City"].astype(str).str.lower() == low]["Store ID"].tolist()
    if not ids:
        ids = stores[stores["Store"].astype(str).str.lower().str.contains(re.escape(low))]["Store ID"].tolist()
    if not ids:
        return None, {"error": f"Unknown store '{value}'.", "valid_values": sorted(stores["Store"].astype(str).unique())}
    if len(ids) > 1:
        labels = stores[stores["Store ID"].isin(ids)]["Store"].tolist()
        return None, {
            "error": f"'{value}' matches more than one store ({', '.join(labels)}). Please specify the Store ID.",
            "valid_values": sorted(labels),
        }
    return ids[0], None


def _subset(fact, year=None, channel=None, category=None, store=None):
    """Apply single-value filters. Returns (subset, applied_filters, error_or_None)."""
    applied = {}
    d = fact
    if year is not None:
        years = sorted(int(y) for y in fact["Year"].dropna().unique())
        if int(year) not in years:
            return d, applied, {"error": f"Year {year} is not in the loaded data.", "valid_values": years}
        d = d[d["Year"] == int(year)]
        applied["year"] = int(year)
    if channel is not None:
        ch, err = _match_option(channel, fact["Channel"].unique(), "channel")
        if err:
            return d, applied, err
        d = d[d["Channel"] == ch]
        applied["channel"] = ch
    if category is not None:
        ca, err = _match_option(category, fact["Category"].unique(), "category")
        if err:
            return d, applied, err
        d = d[d["Category"] == ca]
        applied["category"] = ca
    if store is not None:
        sid, err = _match_store(fact, store)
        if err:
            return d, applied, err
        d = d[d["Store ID"] == sid]
        applied["store"] = d["Store"].iloc[0] if len(d) else sid
    return d, applied, None


def _pct_change(old, new):
    if old is None or new is None or old == 0 or pd.isna(old) or pd.isna(new):
        return None
    return 100 * (new - old) / old


def _partial_year_notes(fact, years) -> list[str]:
    out = []
    for y in years:
        n = months_present(fact, int(y))
        if n < 12:
            out.append(f"{int(y)} has only {n} month(s) of data, so comparisons involving it are not like-for-like.")
    return out


def get_total_revenue(fact, year=None, channel=None, category=None, store=None) -> dict:
    """Total revenue for the data, optionally narrowed by year / channel / category / store."""
    d, applied, err = _subset(fact, year, channel, category, store)
    if err:
        return err
    out = {
        "total_revenue": round(total_revenue(d), 2),
        "rows_summed": int(len(d)),
        "filters_applied": applied or "none",
        "period_covered": period_label(d),
    }
    if len(d) == 0:
        out["note"] = "No rows match these filters."
    nulls = int(d["Revenue"].isna().sum())
    if nulls:
        out["warning"] = f"{nulls} row(s) have missing revenue and were not counted."
    return _clean(out)


def get_revenue_by_year(fact, channel=None, category=None, store=None) -> dict:
    """Revenue for each calendar year with year-over-year change."""
    d, applied, err = _subset(fact, None, channel, category, store)
    if err:
        return err
    t = revenue_by_year(d)
    if t.empty:
        return {"note": "No dated revenue rows match.", "filters_applied": applied or "none"}
    t["Months of data"] = t["Year"].map(lambda y: months_present(d, y))
    t["YoY change"] = t["Revenue"].diff()
    t["YoY change %"] = t["Revenue"].pct_change() * 100
    notes = _partial_year_notes(d, t["Year"])
    return _clean(
        {
            "by_year": _records(t.rename(columns={"Revenue": "revenue"})),
            "total_revenue": round(float(t["Revenue"].sum()), 2),
            "filters_applied": applied or "none",
            "notes": notes,
        }
    )


def get_revenue_by_channel(fact, year=None, category=None) -> dict:
    """Revenue split by sales channel (e.g. Website vs Brick & Mortar), with growth between first and last year."""
    d, applied, err = _subset(fact, year, None, category, None)
    if err:
        return err
    if d.empty:
        return {"note": "No rows match.", "filters_applied": applied or "none"}
    by_ch = revenue_by(d, "Channel").rename(columns={"Channel": "channel", "Revenue": "revenue", "Share %": "share_pct"})
    result = {"by_channel": _records(by_ch), "filters_applied": applied or "none", "notes": []}
    yc = revenue_by_year(d, "Channel")
    result["by_year_and_channel"] = _records(yc.rename(columns={"Channel": "channel", "Revenue": "revenue"}))
    years = sorted(yc["Year"].unique())
    if len(years) >= 2:
        first, last = years[0], years[-1]
        piv = yc.pivot(index="Channel", columns="Year", values="Revenue").fillna(0.0)
        growth = []
        for ch in piv.index:
            a, b = float(piv.loc[ch, first]), float(piv.loc[ch, last])
            growth.append(
                {"channel": ch, "first_year": int(first), "last_year": int(last), "revenue_first_year": a,
                 "revenue_last_year": b, "change": b - a, "pct_change": _pct_change(a, b)}
            )
        growth.sort(key=lambda r: r["change"], reverse=True)
        result["growth_first_to_last_year"] = growth
        result["greatest_absolute_growth"] = growth[0]["channel"]
        pct_ok = [g for g in growth if g["pct_change"] is not None]
        if pct_ok:
            result["greatest_pct_growth"] = max(pct_ok, key=lambda g: g["pct_change"])["channel"]
        result["notes"] += _partial_year_notes(d, [first, last])
        result["notes"].append("'Growth' can mean dollars or percent; both are shown and they can name different channels.")
    return _clean(result)


def get_monthly_revenue_share(fact, year=None, channel=None, category=None, store=None) -> dict:
    """Each month's revenue as a % of its own calendar year's revenue, plus the average share per month."""
    d, applied, err = _subset(fact, year, channel, category, store)
    if err:
        return err
    ms = monthly_share(d)
    if ms.empty:
        return {"note": "No dated revenue rows match.", "filters_applied": applied or "none"}
    full_years = [y for y in ms["Year"].unique() if months_present(d, y) == 12]
    avg = None
    if full_years:
        avg = (
            ms[ms["Year"].isin(full_years)].groupby(["Month", "Month name"], as_index=False)["Share of year %"].mean()
            .rename(columns={"Share of year %": "average_share_pct"})
        )
    out = {
        "monthly": _records(ms.rename(columns={"Revenue": "revenue", "Share of year %": "share_of_year_pct"})),
        "denominator": "that calendar year's total revenue within the same filters",
        "filters_applied": applied or "none",
        "notes": _partial_year_notes(d, ms["Year"].unique()),
    }
    if avg is not None:
        out["average_share_by_month"] = _records(avg)
        out["peak_month"] = avg.loc[avg["average_share_pct"].idxmax(), "Month name"]
        out["lowest_month"] = avg.loc[avg["average_share_pct"].idxmin(), "Month name"]
        out["years_used_for_average"] = [int(y) for y in full_years]
    return _clean(out)


def _store_year_revenue(d: pd.DataFrame) -> pd.DataFrame:
    return d.dropna(subset=["Year"]).groupby(["Store ID", "Store", "Channel", "Year"], as_index=False)["Revenue"].sum()


def get_store_trend(fact, start_year=None, end_year=None, channel=None, category=None, top_n=None) -> dict:
    """Revenue change per location between two years (default: first and last year in the data)."""
    d, applied, err = _subset(fact, None, channel, category, None)
    if err:
        return err
    sy = _store_year_revenue(d)
    if sy.empty:
        return {"note": "No dated revenue rows match.", "filters_applied": applied or "none"}
    years = sorted(int(y) for y in sy["Year"].unique())
    start = int(start_year) if start_year is not None else years[0]
    end = int(end_year) if end_year is not None else years[-1]
    for y in (start, end):
        if y not in years:
            return {"error": f"Year {y} is not in the loaded data.", "valid_values": years}
    if start == end:
        return {"error": "start_year and end_year must differ.", "valid_values": years}
    a = sy[sy["Year"] == start].set_index("Store ID")["Revenue"]
    b = sy[sy["Year"] == end].set_index("Store ID")["Revenue"]
    meta = sy.drop_duplicates("Store ID").set_index("Store ID")[["Store", "Channel"]]
    rows = []
    for sid in meta.index:
        ra, rb = float(a.get(sid, 0.0)), float(b.get(sid, 0.0))
        rows.append(
            {"store_id": sid, "store": meta.loc[sid, "Store"], "channel": meta.loc[sid, "Channel"],
             "revenue_start": ra, "revenue_end": rb, "change": rb - ra, "pct_change": _pct_change(ra, rb)}
        )
    rows.sort(key=lambda r: (r["pct_change"] is None, -(r["pct_change"] or 0)))
    declined = [r for r in rows if r["change"] < 0]
    comparable = [r for r in rows if r["pct_change"] is not None]
    out = {
        "start_year": start, "end_year": end,
        "filters_applied": applied or "none",
        "stores": rows[: int(top_n)] if top_n else rows,
        "stores_compared": len(rows),
        "stores_that_declined": len(declined),
        "largest_pct_decline": min(declined, key=lambda r: r["pct_change"] if r["pct_change"] is not None else 0)
        if any(r["pct_change"] is not None for r in declined) else None,
        "largest_pct_growth": max(comparable, key=lambda r: r["pct_change"]) if comparable else None,
        "notes": _partial_year_notes(d, [start, end]),
    }
    if not declined:
        out["notes"].append("No location declined between these years.")
    no_start = [r["store"] for r in rows if r["revenue_start"] == 0]
    if no_start:
        out["notes"].append(
            "No revenue in the start year (opened later or missing data), so % change is undefined: " + ", ".join(no_start)
        )
    return _clean(out)


def get_store_ranking(fact, year=None, channel=None, category=None, top_n=None) -> dict:
    """Locations ranked by revenue (optionally for one year)."""
    d, applied, err = _subset(fact, year, channel, category, None)
    if err:
        return err
    if d.empty:
        return {"note": "No rows match.", "filters_applied": applied or "none"}
    r = revenue_by(d, "Store").rename(columns={"Store": "store", "Revenue": "revenue", "Share %": "share_pct"})
    r["rank"] = range(1, len(r) + 1)
    if top_n:
        r = r.head(int(top_n))
    return _clean({"ranking": _records(r), "filters_applied": applied or "none", "period_covered": period_label(d)})


def get_category_performance(fact, year=None, channel=None, store=None, level="category") -> dict:
    """Revenue by product category (or by individual product with level='product'), with growth first->last year."""
    d, applied, err = _subset(fact, year, channel, None, store)
    if err:
        return err
    if d.empty:
        return {"note": "No rows match.", "filters_applied": applied or "none"}
    dim = "Product label" if str(level).lower() == "product" else "Category"
    r = revenue_by(d, dim).rename(columns={dim: "name", "Revenue": "revenue", "Share %": "share_pct"})
    out = {"level": dim, "performance": _records(r), "filters_applied": applied or "none", "notes": []}
    yc = revenue_by_year(d, dim)
    years = sorted(yc["Year"].unique())
    if len(years) >= 2:
        first, last = years[0], years[-1]
        piv = yc.pivot(index=dim, columns="Year", values="Revenue").fillna(0.0)
        growth = [
            {"name": n, "first_year": int(first), "last_year": int(last), "revenue_first_year": float(piv.loc[n, first]),
             "revenue_last_year": float(piv.loc[n, last]), "change": float(piv.loc[n, last] - piv.loc[n, first]),
             "pct_change": _pct_change(float(piv.loc[n, first]), float(piv.loc[n, last]))}
            for n in piv.index
        ]
        growth.sort(key=lambda g: g["change"], reverse=True)
        out["growth_first_to_last_year"] = growth
        out["notes"] += _partial_year_notes(d, [first, last])
    return _clean(out)


def get_satisfaction_vs_revenue(fact, survey) -> dict:
    """
    Association between survey Satisfaction/Interest and annual revenue across location-years.
    This is correlation only - it cannot show that satisfaction causes revenue (or the reverse).
    """
    sy = fact.dropna(subset=["Year"]).groupby(["Store ID", "Year"], as_index=False)["Revenue"].sum()
    if sy.empty:
        return {"note": "No dated revenue rows."}
    m = sy.merge(survey, on=["Store ID", "Year"], how="left")
    with_sat = m.dropna(subset=["Satisfaction"])
    missing = m[m["Satisfaction"].isna()]
    out = {
        "location_years_in_scope": int(len(m)),
        "location_years_with_satisfaction": int(len(with_sat)),
        "location_years_missing_satisfaction": int(len(missing)),
        "locations_missing_all_satisfaction": sorted(
            set(missing["Store ID"]) - set(with_sat["Store ID"])
        ),
        "caveats": [
            "Correlation is not causation: stores differ in size, market and location, and revenue can influence "
            "satisfaction as easily as the reverse.",
            "Per the survey workbook's 'Terms and Definitions' sheet, Satisfaction is a 0-5 rating of the shopper's experience at their "
            "local store and 'Interest' is a 0-100 rating of how much respondents value quality espresso products (not interest in E-Presso itself). "
            "The number of respondents and response rates are not in the data.",
            "Locations without survey responses (such as the website) are excluded from the correlation.",
            "Large stores (high revenue) and small stores may differ systematically, which can drive the pattern.",
        ],
    }
    if len(with_sat) >= 3:
        out["pearson_correlation_satisfaction_vs_revenue"] = float(with_sat["Satisfaction"].corr(with_sat["Revenue"]))
        out["spearman_rank_correlation_satisfaction_vs_revenue"] = float(
            with_sat["Satisfaction"].rank().corr(with_sat["Revenue"].rank())
        )
        inter = m.dropna(subset=["Interest"])
        if len(inter) >= 3:
            out["pearson_correlation_interest_vs_revenue"] = float(inter["Interest"].corr(inter["Revenue"]))
    else:
        out["note"] = "Too few location-years with satisfaction scores to compute a correlation."
    return _clean(out)


def get_data_quality_report(ds: Dataset) -> dict:
    """Missing values, duplicates, orphan keys and other potential data problems in the loaded data."""
    p = profile_dataset(ds)
    return _clean(
        {
            "table_summary": _records(p["summary"]),
            "missing_values": _records(p["missing"]),
            "relationship_checks": _records(p["relationships"]),
            "other_checks": p["checks"],
            "join_reconciliation": _records(p["reconciliation"]),
            "warnings": p["notes"],
        }
    )


def get_data_limitations(ds: Dataset) -> dict:
    """What the loaded data can and cannot support (e.g. whether profit can be computed)."""
    cols = {name: [str(c) for c in df.columns] for name, df in ds.tables.items()}
    cost_words = ("cost", "profit", "margin", "expense", "cogs", "price", "quantity", "units", "rent", "lease", "wage")
    cost_like = sorted(f"{t}.{c}" for t, cs in cols.items() for c in cs if any(w in c.lower() for w in cost_words))
    return _clean(
        {
            "columns_by_table": cols,
            "cost_or_profit_like_columns_found": cost_like or "none",
            "measure_available": "Revenue (a stored dollar amount per store/product/month)",
            "not_available": [
                "costs, expenses, operating or lease costs, margins or profit (revenue is NOT profit)",
                "units sold or prices (revenue cannot be split into quantity x price)",
                "customer-level data (survey is per location per year, not per customer)",
                "number of survey respondents or response rates (the survey is a per-location annual summary)",
                "reasons for changes (the data cannot establish causes)",
            ],
            "profit_tool_available": False,
        }
    )
