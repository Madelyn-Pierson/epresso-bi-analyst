"""Main app. Stage 1: upload ONE OR MORE files (CSV / TXT / Excel / Access / SQLite),
pick the table to analyze, and see a profile of it. Nothing here assumes a particular dataset.
Keep calculations in analytics.py, file reading in loading.py, and the assistant in agent.py."""
import streamlit as st

import agent
import analytics
import loading
import pandas as pd

MAX_QUESTIONS = 15  # per-session limit for the assistant

st.set_page_config(page_title='E-Presso BI Analyst', layout='wide')
st.title('E-Presso BI Analyst')
st.caption('Starter app. Replace this line with a description of your own application.')


@st.cache_data(show_spinner='Reading file...')
def cached_load(file_bytes: bytes, filename: str):
    # Cached per FILE and keyed on its contents: adding a second file never re-reads the
    # first, and an edited file is never confused with the original.
    return loading.load_file(file_bytes, filename)


# ---------------------------------------------------------------- 1. upload
with st.sidebar:
    st.header('Data')
    uploads = st.file_uploader(
        'Upload one or more files',
        type=loading.SUPPORTED_TYPES,
        accept_multiple_files=True,
        help='CSV / TXT / TSV, Excel, Access (.accdb, .mdb) or SQLite. One file is enough. '
             'Files with several sheets or tables are split into separate tables.',
    )

if not uploads:
    st.info('Upload one or more files in the sidebar to begin.')
    st.stop()

# ---------------------------------------------------------------- 2. read every file
loaded: dict[str, loading.LoadedTable] = {}
for f in uploads:
    try:
        result = cached_load(f.getvalue(), f.name)
    except loading.LoadError as exc:  # one bad file must not block the others
        st.error(f'Could not read {f.name}: {exc}')
        continue
    for warning in result.warnings:
        st.warning(f'{f.name}: {warning}')
    for table in result.tables:
        label, n = table.label, 2
        while label in loaded:  # the same file uploaded twice
            label, n = f'{table.label} ({n})', n + 1
        loaded[label] = table

usable = {label: t for label, t in loaded.items() if t.usable}
if not usable:
    st.error('No readable tables were found in the uploaded files.')
    st.stop()

# ---------------------------------------------------------------- 3. choose the table to analyze
names = list(usable)
with st.sidebar:
    st.subheader('Table to analyze')
    if len(names) == 1:
        chosen = names[0]
        st.caption(f'Using: {chosen}')
    else:
        biggest = max(names, key=lambda n: len(usable[n].df))
        chosen = st.selectbox('Working table', names, index=names.index(biggest),
                              help='Everything on the other tabs uses this table.')
df = usable[chosen].df

# ---------------------------------------------------------------- 4. tabs
tab_data, tab_dash, tab_assistant = st.tabs(['Data and profile', 'Dashboard', 'Assistant'])

with tab_data:
    st.subheader('Files and tables found')
    st.dataframe(loading.describe_tables(list(loaded.values())), hide_index=True)

    st.subheader(f'Profile: {chosen}')
    profile = analytics.profile_table(df)
    c1, c2, c3 = st.columns(3)
    c1.metric('Rows', f"{profile['rows']:,}")
    c2.metric('Columns', profile['columns'])
    c3.metric('Duplicate rows', profile['duplicate_rows'])
    st.subheader('Preview')
    st.dataframe(df.head(50))
    left, right = st.columns(2)
    left.subheader('Missing values')
    left.dataframe(profile['missing'])
    right.subheader('Data types')
    right.dataframe(profile['dtypes'])


with tab_dash:
    st.header("Business Performance Dashboard")
    st.caption(
        "Select the columns that describe your business data. "
        "Metrics and charts update when filters change."
    )

    data = df.copy()

    if data.empty or len(data.columns) == 0:
        st.warning("The selected table has no usable data.")
        st.stop()

    columns = data.columns.tolist()

    # ---------- Column configuration ----------
    st.subheader("1. Configure your metrics")

    numeric_cols = data.select_dtypes(include="number").columns.tolist()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_col = st.selectbox(
            "Numeric measure (e.g., revenue)",
            ["None"] + numeric_cols,
            key="dashboard_metric",
        )

    with c2:
        date_col = st.selectbox(
            "Date column (optional)",
            ["None"] + columns,
            key="dashboard_date",
        )

    with c3:
        group_col = st.selectbox(
            "Category or grouping column",
            ["None"] + columns,
            key="dashboard_group",
        )

    if metric_col == "None":
        st.info("Choose a numeric measure to calculate metrics and charts.")
        st.stop()

    # ---------- Date preparation ----------
    if date_col != "None":
        parsed_dates = pd.to_datetime(data[date_col], errors="coerce")
        valid_dates = parsed_dates.notna()

        if valid_dates.any():
            data[date_col] = parsed_dates
            data["_dashboard_year"] = parsed_dates.dt.year
        else:
            st.warning(
                f"Column '{date_col}' could not be interpreted as dates. "
                "Date filtering and time-series charts are unavailable."
            )
            date_col = "None"

    # ---------- Interactive filters ----------
    st.subheader("2. Filter the data")

    filtered = data.copy()

    if date_col != "None":
        years = sorted(
            filtered["_dashboard_year"].dropna().unique().tolist()
        )
        if years:
            selected_years = st.multiselect(
                "Year",
                years,
                default=years,
                key="dashboard_year_filter",
            )
            filtered = filtered[
                filtered["_dashboard_year"].isin(selected_years)
            ]

    # Add filters for columns with a manageable number of categories.
    filter_cols = [
        col for col in columns
        if col != date_col
        and col != metric_col
        and data[col].nunique(dropna=True) <= 50
        and data[col].nunique(dropna=True) > 1
    ]

    selected_filter_cols = st.multiselect(
        "Choose additional filter dimensions",
        filter_cols,
        key="dashboard_filter_dimensions",
        help="For example, product category, store, or sales channel.",
    )

    if selected_filter_cols:
        filter_layout = st.columns(min(3, len(selected_filter_cols)))

        for i, col in enumerate(selected_filter_cols):
            options = sorted(
                filtered[col].dropna().unique().tolist(),
                key=str,
            )

            with filter_layout[i % len(filter_layout)]:
                selected = st.multiselect(
                    col,
                    options,
                    default=options,
                    key=f"dashboard_filter_{col}",
                )

            filtered = filtered[filtered[col].isin(selected)]

    st.caption(
        f"{len(filtered):,} of {len(data):,} rows match the filters."
    )

    if filtered.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    # ---------- KPI cards ----------
    st.subheader("3. Key metrics")

    values = pd.to_numeric(filtered[metric_col], errors="coerce")
    total = values.sum()
    valid_values = values.dropna()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(f"Total {metric_col}", f"{total:,.2f}")
    k2.metric("Average per row", f"{valid_values.mean():,.2f}"
              if not valid_values.empty else "N/A")
    k3.metric("Valid numeric records", f"{len(valid_values):,}")
    k4.metric("Rows included", f"{len(filtered):,}")

    # ---------- Visualization 1: trend over time ----------
    st.subheader("4. Visualizations")

    if date_col != "None":
        trend_data = filtered.dropna(subset=[date_col]).copy()
        trend_data[metric_col] = pd.to_numeric(
            trend_data[metric_col], errors="coerce"
        )

        trend = (
            trend_data.set_index(date_col)[metric_col]
            .resample("MS")
            .sum()
            .rename(f"Total {metric_col}")
        )

        st.markdown("**Chart 1 — Monthly trend**")
        st.line_chart(trend, y=f"Total {metric_col}")
    else:
        st.info(
            "Chart 1: Select a date column to display a monthly trend."
        )

    # ---------- Visualization 2: breakdown ----------
    if group_col != "None":
        breakdown_data = filtered.copy()
        breakdown_data[metric_col] = pd.to_numeric(
            breakdown_data[metric_col], errors="coerce"
        )

        breakdown = (
            breakdown_data.dropna(subset=[group_col, metric_col])
            .groupby(group_col)[metric_col]
            .sum()
            .sort_values(ascending=False)
        )

        st.markdown(f"**Chart 2 — {metric_col} by {group_col}**")
        st.bar_chart(breakdown, y=metric_col)
    else:
        st.info(
            "Chart 2: Select a grouping column to compare categories."
        )

    # ---------- Visualization 3: distribution ----------
    st.markdown(f"**Chart 3 — Distribution of {metric_col}**")
    st.bar_chart(
        valid_values.value_counts(bins=10, sort=False)
        .rename("Number of records"),
        y="Number of records",
    )

    # ---------- Data quality ----------
    with st.expander("Data quality for the selected table"):
        profile = analytics.profile_table(data)

        q1, q2, q3 = st.columns(3)
        q1.metric("Source rows", f"{profile['rows']:,}")
        q2.metric("Columns", profile["columns"])
        q3.metric("Duplicate rows", profile["duplicate_rows"])

        st.markdown("**Missing values by column**")
        st.dataframe(profile["missing"], hide_index=True)

        st.markdown("**Data types**")
        st.dataframe(profile["dtypes"], hide_index=True)

        st.markdown("**Filtered records**")
        st.dataframe(filtered, use_container_width=True, hide_index=True)


with tab_assistant:
    st.caption('Mode: rules-based starter. Describe what you actually built here.')
    st.session_state.setdefault('n_questions', 0)
    st.session_state.setdefault('history', [])
    for item in st.session_state['history']:
        with st.chat_message('user'):
            st.write(item['question'])
        with st.chat_message('assistant'):
            st.markdown(f"**Scope:** {item['answer']['scope']}")
            st.markdown(f"**Findings:** {item['answer']['findings']}")
            if item['answer']['interpretation']:
                st.markdown(f"**Interpretation:** {item['answer']['interpretation']}")
            with st.expander('Tool-call log'):
                st.json(item['answer']['tool_log'])
    question = st.chat_input('Ask a question about the loaded data')
    if question:
        if st.session_state['n_questions'] >= MAX_QUESTIONS:
            st.warning('Question limit reached for this session. Reload the page to start over.')
        else:
            st.session_state['n_questions'] += 1
            answer = agent.answer_question(question, df, mode='rules')
            st.session_state['history'].append({'question': question, 'answer': answer})
            st.rerun()