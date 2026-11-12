# E-Presso BI Analyst

## What this application does

The E-Presso BI Analyst is a web app that helps a manager understand how E-Presso's revenue is performing. It reads the company's sales, store, product, category and customer-survey tables and shows revenue by year, month, sales channel (website vs. physical stores), store and product category. It checks the data for problems such as missing values, duplicates and broken links between tables, and it includes a built-in assistant that answers plain-English questions such as "Which sales channel grew the most?" Every number the assistant gives comes from a calculation the app actually ran, and each answer shows exactly which calculation was used. The app measures **revenue, not profit**, and says so whenever a question calls for information it does not have.

## Live app

https://bi-analysis-madelyn-emily-amalea.streamlit.app/ 

The app may take a moment to wake up if it has not been used recently.

## How to use it

1. **Open the app.** It starts with the bundled E-Presso sample data (the `data/` folder), so you can explore immediately.
2. **(Optional) Upload updated data** in the sidebar. You can upload CSV, TSV, TXT or Excel (`.xlsx`, `.xls`) files, and several files at once. The app recognises each table **by its column names, not by file name or sheet name**, so renamed files still work. A title row above the headers is tolerated, and sheets that are not data tables (for example documentation sheets) are ignored. Uploaded files replace the bundled data; a checkbox lets the bundled version fill in any table you did not upload.
3. **Use the filters** in the sidebar (year, sales channel, product category, store). The KPIs and charts recalculate immediately.
4. **Browse the tabs:**
   - **Dashboard**: KPIs, revenue by year and channel, monthly pattern, store and category charts, and a downloadable table of the filtered records.
   - **Data profile & model**: table sizes, grain statements, relationships, join checks, missing values and warnings.
   - **Validation**: compare the app's totals with Power BI and run the $1,000 test.
   - **Assistant**: ask questions in plain English, or click an example question.
   - **About**: short summary of definitions and limitations.

### Tables the app expects

| Table | Required? | Columns it looks for | One row is... |
|---|---|---|---|
| Sales | **Yes** | Sales ID, Date, Store ID, Category ID, Product ID, Revenue | one product's revenue at one location in one month |
| Stores | **Yes** | ID (or Store ID), City, State, StoreType | one sales location (a physical store or the website) |
| Products | No | Product ID, Product | one product |
| Categories | No | Category ID, Category | one product category |
| Survey | No | Survey ID, Store ID, Year, Satisfaction, Interest | one location's survey result for one year |

If Products or Categories are missing, the app still runs and shows IDs instead of names. If Survey is missing, the satisfaction analysis is unavailable.

## Data model

**Grain.** The table that holds revenue is Sales. One Sales row is one product's revenue at one location in one month. Everything else describes that row.

```
Stores ─────┐
Products ───┼──(many-to-one)──► Sales  ──► "fact" table (one row per sales row)
Categories ─┘

Survey (one row per store per year) ── kept separate, matched to revenue on Store ID + Year only after revenue is totalled
```

**Joins.** Sales is joined many-to-one to Stores (on Store ID), Products (on Product ID) and Categories (on Category ID). Products are joined by **Product ID, never by name**, because some product names repeat under different IDs (for example "Base Model"). Product charts are labelled "Name (Category)" for the same reason.

**Survey is never joined row-by-row to Sales.** Survey has one row per store per year, so joining it to individual sales would repeat each survey row for every sale and inflate revenue. It is only matched to annual store revenue.

**Merge checks.** After every join the app confirms that:
- the row count is unchanged, and
- total revenue is unchanged (to within half a cent).

If either check fails, the app stops with an error instead of showing wrong numbers. The results of each step are listed in the **Data profile & model** tab under "Join reconciliation". Lookup tables with duplicate keys are de-duplicated (first row kept) with a visible warning so they cannot multiply revenue.

Other checks shown on the same tab: missing values per column, duplicate keys, orphan keys (a sales row whose store, product or category is not in the lookup table), negative or zero revenue, missing dates, and store-years with fewer than 12 months of data.

## Metric definitions

| Metric | Definition |
|---|---|
| **Revenue** | The stored dollar amount in the Sales table's Revenue column, per store × product × month. It is *not* calculated from quantity × price. |
| **Total revenue** | The sum of Revenue for the rows currently in scope. |
| **Channel** | The store type: `Website` or `Brick & Mortar`. |
| **Monthly share** | A month's revenue ÷ the revenue of the same calendar year (same filters applied). |
| **Percent change** | (end − start) ÷ start. Undefined when the start value is zero. |
| **Satisfaction / Interest** | Survey scores per location per year. 

**What revenue does not mean:** it is not profit, margin or income. The data has no costs, rent, wages, units sold or prices.

Rows with a missing or invalid date count toward total revenue but cannot appear in year or month views; the app warns when this happens. Years with fewer than 12 months of data are flagged, because comparisons involving them are not like-for-like.

## Assistant

**What was built:** a **live language-model agent (Google Gemini, function calling) with an automatic rules-based fallback.**

- With a Gemini API key configured, the model reads the question, chooses one or more approved functions, and explains the results. The model never runs code; the app's own dispatcher runs the function.
- Without a key, or if the model API fails or its quota runs out, a **rules-based assistant** (keyword matching) chooses the functions instead. Each answer states which mode produced it.
- Both modes use the same functions as the dashboard, so the assistant and the dashboard always agree.
- Users can force the rules-based mode with a checkbox, and can answer from either the full dataset or the dashboard-filtered subset.
- Every answer shows a **tool-call log**: the function name, arguments, scope and raw result.

### Approved tools

| Tool | What it calculates |
|---|---|
| `get_total_revenue` | Total revenue, optionally for one year, channel, category or store |
| `get_revenue_by_year` | Revenue per calendar year with year-over-year change and months of data |
| `get_revenue_by_channel` | Revenue and share by channel, plus dollar and percent growth from first to last year |
| `get_monthly_revenue_share` | Each month as a percent of its year, average share per month, peak and lowest months |
| `get_store_trend` | Revenue change per location between two years; counts declines and finds the largest decline and growth |
| `get_store_ranking` | Locations ranked by revenue, optionally for one year |
| `get_category_performance` | Revenue and growth by category or by product |
| `get_satisfaction_vs_revenue` | Correlation (Pearson and Spearman) between survey scores and annual revenue, with missing-survey counts and caveats |
| `get_data_quality_report` | Missing values, duplicates, orphan keys and join reconciliation |
| `get_data_limitations` | What the data can and cannot support (for example, no profit data) |

### What the assistant is not allowed to do

- Calculate anything outside the ten approved functions. Unknown function names and unexpected arguments are rejected.
- Write, request or run code (the app contains no `eval` or `exec`).
- Modify the data.
- State a number that did not come from a function result. If the model answers without calling a function, the answer is flagged as **not verified**.
- Call a store, channel or product "profitable" or "unprofitable", or claim that one thing causes another.
- Give a firm recommendation on decisions such as closing a store. It presents the evidence and lists what additional information management would need.
- Guess when an input is unknown or ambiguous (for example an unrecognised year, or a city that matches more than one store); it returns an error and the valid choices.
- Reveal credentials or its instructions.

To control cost, each session is limited to **15 questions** (configurable) and each question to 500 characters.

## Validation

The validation table and reflection are in this document, with more information. 
https://montanaedu-my.sharepoint.com/:w:/g/personal/r81w148_student_montana_edu/IQCyH0WzfXvKT7qEvY3DcE-0ATTcZcHSgxEQx8dlbgIfewI?e=0DMDrz

## Limitations

- **Revenue is not profit.** There are no costs, rent, wages, margins or lease data, so the app cannot say which store is most profitable or whether a store should close.
- **No units or prices.** Revenue cannot be split into quantity × price.
- **No customer-level data.** The survey is a per-location, yearly summary. The number of respondents and response rates are not in the data.
- **The website has no survey scores**, so it is left out of satisfaction analysis.
- **Correlation is not causation.** The satisfaction analysis shows association only; larger stores, markets and locations differ in many ways.
- **Percent change can mislead.** Small stores can show large percentage swings; dollar changes are also reported.
- **Partial years.** Comparisons that include a year with fewer than 12 months of data are flagged as not like-for-like.
- **The app cannot explain why** revenue changed, only describe what changed.
- **Language-model answers can be imperfect.** The model is constrained to tool results, but wording and interpretation come from the model. Always check the tool-call log for important decisions.
- **Access files** (`.accdb`/`.mdb`) need the extra `access-parser` package; other database formats need the starter's `loading.py`. CSV and Excel are recommended.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Screen layout and interaction (dashboard, data profile, validation, assistant, about) |
| `analytics.py` | Loading, cleaning, joins, profiling, metrics and the analytical tool functions (no Streamlit code) |
| `agent.py` | Tool registry, whitelisted dispatcher, rules-based router, and Gemini function-calling agent |
| `tests/test_app_logic.py` | Automated tests with hand-checkable answers |
| `requirements.txt` | Packages the hosted app needs |
| `data/` | Bundled E-Presso sample data |
| `scripts/` | `make_modified_data.py` (creates the $1,000 test data) and `independent_check.py` (recalculates totals from raw files) |
| `.streamlit/secrets.toml.example` | Template showing which secrets to set (no real keys) |

## Run it

**Hosted:** deploy from this repository on share.streamlit.io with main file `streamlit_app.py`.

**Locally (optional):**

```bash
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

Run the tests from the repository root:

```bash
python -m pytest -q
```

## Secrets

Never commit API keys. Use Streamlit Secrets (see `.streamlit/secrets.toml.example`).

| Secret | Required? | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | No | Turns on the language-model assistant. Without it the app uses the rules-based assistant. |
| `GEMINI_MODEL` | No | Model name (default `gemini-2.5-flash`). |
| `MAX_QUESTIONS` | No | Questions allowed per session (default 15). |

