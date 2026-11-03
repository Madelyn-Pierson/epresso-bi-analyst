"""Main app.
Flow: upload files -> profile -> MODEL (join tables, define revenue, pick the date) ->
DASHBOARD (revenue KPIs, filters, charts) -> ASSISTANT.

Keep calculations in analytics.py, file reading in loading.py, and the assistant in agent.py.
The dashboard calls the SAME functions as the assistant, so what you validate against
Power BI is exactly what the assistant will later report."""
import calendar
import re

import pandas as pd
import streamlit as st

import agent
import analytics
import loading

MAX_QUESTIONS = 15  # per-session limit for the assistant

st.set_page_config(page_title='E-Presso BI Analyst', layout='wide')
st.title('E-Presso BI Analyst')
st.caption('Upload your sales, store, product and survey files, build the data model, '
           'then explore revenue with filters.')


@st.cache_data(show_spinner='Reading file...')
def cached_load(file_bytes: bytes, filename: str):
    # Cached per FILE and keyed on its contents: adding a second file never re-reads the
    # first, and an edited file is never confused with the original.
    return loading.load_file(file_bytes, filename)


def money(x) -> str:
    return f'{x:,.2f}'


# ---------------------------------------------------------------- 1. upload
with st.sidebar:
    st.header('Data')
    uploads = st.file_uploader(
        'Upload one or more files',
        type=loading.SUPPORTED_TYPES,
        accept_multiple_files=True,
        help='CSV / TXT / TSV, Excel, Access (.accdb, .mdb) or SQLite. '
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


# ---------------------------------------------------------------- 3. model tab
def render_model_tab(usable: dict):
    """Build the analysis table from the loaded tables. Returns a dict (the model) or None.
    No st.stop() here: stopping inside a tab would also hide the other tabs."""
    st.header('Build the data model')
    st.caption('Revenue usually lives in one big table of transactions (the FACT table) while names '
               'such as store type, category and product live in small LOOKUP tables. '
               'Joining them lets you slice revenue by those names.')

    names = list(usable)
    biggest = max(names, key=lambda n: len(usable[n].df))
    fact_name = st.selectbox('1. Fact table (the one with one row per sale or transaction)', names,
                             index=names.index(biggest), key='model_fact')
    fact = usable[fact_name].df
    st.caption(f'{len(fact):,} rows. Every join below must keep exactly this many rows.')

    # ---- joins
    st.subheader('2. Lookup tables')
    merged = fact.copy()
    others = [n for n in names if n != fact_name]
    suggestions = {n: analytics.suggest_join_keys(fact, usable[n].df) for n in others}
    others.sort(key=lambda n: 0 if suggestions[n] else 1)  # tables with an obvious key first
    guess_value_col = analytics.suggest_revenue_column(fact)
    join_reports = []

    if not others:
        st.info('Only one table was uploaded, so there is nothing to join. '
                'Upload the lookup tables too if you want to slice by store, category, etc.')
    for n in others:
        best = suggestions[n][0] if suggestions[n] else None
        title = f'{n}  (key found: {best["left"]} = {best["right"]})' if best else f'{n}  (no obvious key)'
        with st.expander(title, expanded=bool(best)):
            use = st.checkbox('Join this table to the model', value=bool(best), key=f'model_use_{fact_name}|{n}')
            if not use:
                continue
            right_df = usable[n].df
            left_opts, right_opts = list(merged.columns), list(right_df.columns)
            c1, c2 = st.columns(2)
            left_key = c1.selectbox('Key column in the model so far', left_opts,
                                    index=left_opts.index(best['left']) if best and best['left'] in left_opts else 0,
                                    key=f'model_lk_{fact_name}|{n}')
            right_key = c2.selectbox(f'Key column in this table', right_opts,
                                     index=right_opts.index(best['right']) if best else 0,
                                     key=f'model_rk_{fact_name}|{n}')
            try:
                merged, rep = analytics.join_tables(merged, right_df, left_key, right_key,
                                                    right_name=usable[n].name, value_col=guess_value_col)
            except analytics.JoinError as exc:
                st.error(f'Join refused: {exc}')
                continue
            join_reports.append(rep)
            if rep['unmatched_rows']:
                st.warning(f"{rep['unmatched_rows']:,} rows ({rep['unmatched_pct']}%) have no match in this table. "
                           f"Example keys: {rep['unmatched_key_examples']}. "
                           'Check for typos, extra spaces, or missing lookup rows.')
            else:
                st.success(f"All {rep['rows_after']:,} rows matched. Row count unchanged.")
            st.caption('Columns added: ' + ', '.join(rep['added_columns']))
            if rep['lookup_rows_never_used']:
                st.caption(f"{rep['lookup_rows_never_used']} row(s) of this table are never used by the fact table.")

    # ---- revenue
    st.subheader('3. Revenue')
    mode = st.radio('How is revenue defined?',
                    ['Use an existing column', 'Calculate quantity x price (x discount)'],
                    horizontal=True, key='model_rev_mode')
    numeric_cols = analytics.find_numeric_columns(merged)
    if not numeric_cols:
        st.error('No numeric columns were found, so revenue cannot be defined.')
        return None

    if mode == 'Use an existing column':
        default = guess_value_col if guess_value_col in numeric_cols else numeric_cols[0]
        measure = st.selectbox('Revenue column', numeric_cols, index=numeric_cols.index(default),
                               key='model_rev_col')
        coerced = pd.to_numeric(merged[measure], errors='coerce')
        lost = int(coerced.isna().sum() - merged[measure].isna().sum())
        if lost:
            st.warning(f'{lost:,} value(s) in {measure} are not numbers and will be ignored.')
        merged[measure] = coerced
        definition = f'Revenue = sum of column {measure}'
    else:
        q, p, d = st.columns(3)
        qty = q.selectbox('Quantity column', numeric_cols, key='model_qty')
        price = p.selectbox('Unit price column', numeric_cols, index=min(1, len(numeric_cols) - 1), key='model_price')
        disc = d.selectbox('Discount column (fraction, optional)', ['None'] + numeric_cols, key='model_disc')
        measure = 'Revenue (calculated)' if 'Revenue' in merged.columns else 'Revenue'
        try:
            merged, rev_rep = analytics.add_revenue(merged, qty, price, None if disc == 'None' else disc, name=measure)
        except ValueError as exc:
            st.error(str(exc))
            return None
        definition = f"Revenue = {rev_rep['definition']}"
        if rev_rep['rows_without_revenue']:
            st.warning(f"{rev_rep['rows_without_revenue']:,} rows lack a quantity or price, so they have no revenue.")

    # ---- date
    st.subheader('4. Date')
    date_options = ['None'] + analytics.find_date_columns(merged)
    date_choice = st.selectbox('Date column (only columns that really contain dates are listed)',
                               date_options, index=1 if len(date_options) > 1 else 0, key='model_date')
    date_col, year_col, month_col = None, None, None
    if date_choice != 'None':
        merged, drep = analytics.add_date_parts(merged, date_choice)
        date_col, year_col, month_col = date_choice, drep['year_col'], drep['month_col']
        st.caption(f"Dates run from {drep['first']} to {drep['last']}.")
        if drep['unparseable']:
            st.warning(f"{drep['unparseable']:,} value(s) are not valid dates and are left out of time charts.")

    # ---- summary and checks
    st.subheader('Model summary')
    total = analytics.get_total(merged, measure)
    m1, m2, m3 = st.columns(3)
    m1.metric('Rows', f'{len(merged):,}')
    m2.metric('Columns', merged.shape[1])
    m3.metric(f'Total {measure}', money(total.value))
    st.markdown(f'**Definition:** {definition}')

    problems = False
    if len(merged) != len(fact):
        st.error(f'Row count changed from {len(fact):,} to {len(merged):,}. Fix the joins before trusting any total.')
        problems = True
    if measure in fact.columns:
        fact_total = float(pd.to_numeric(fact[measure], errors='coerce').sum())
        if abs(fact_total - total.value) > 1e-6:
            st.error(f'Total changed after joining: {money(fact_total)} before, {money(total.value)} after.')
            problems = True
        else:
            st.success(f'Check passed: the total ({money(total.value)}) equals the fact table total before any join.')
    if any(r['unmatched_rows'] for r in join_reports):
        st.warning('Some rows did not match a lookup table. Their category/store fields will be blank in the dashboard.')
    with st.expander('Preview of the model table'):
        st.dataframe(merged.head(100))

    return {'df': merged, 'measure': measure, 'date_col': date_col, 'year_col': year_col,
            'month_col': month_col, 'definition': definition, 'ok': not problems}


# ---------------------------------------------------------------- 4. dashboard tab
def render_dashboard(model: dict):
    """Revenue dashboard. Uses `return` (not st.stop) so other tabs still render."""
    data, measure, date_col = model['df'], model['measure'], model['date_col']
    st.header('Revenue dashboard')
    st.caption(f"{model['definition']}. Filters are in the sidebar.")
    if not model['ok']:
        st.warning('The model has unresolved problems (see the Model tab). Totals here may be wrong.')

    # ---------- filters (options come from the FULL data so they never shift under you)
    skip = {measure, date_col, model['year_col'], model['month_col']}
    cat_cols = analytics.find_category_columns(data, exclude=skip)
    selections = {}
    with st.sidebar:
        st.divider()
        st.header('Dashboard filters')
        if date_col:
            years = analytics.filter_options(data[model['year_col']])
            chosen = st.multiselect('Year (from the date column)', years, default=years, key='f_year')
            selections[model['year_col']] = chosen
            months = analytics.filter_options(data[model['month_col']])
            chosen = st.multiselect('Month (from the date column)', months, default=months, key='f_month',
                                    format_func=lambda m: calendar.month_abbr[m] if isinstance(m, int) else m)
            selections[model['month_col']] = chosen
        # pre-select up to three filter-friendly columns: few distinct values, and not a
        # year/month field (the date filters above already cover those)
        friendly = [c for c in cat_cols if not re.search(r'(?i)year|month', c)]
        few = sorted(friendly, key=lambda c: data[c].nunique(dropna=True))[:3]
        dims = st.multiselect('Filter by', cat_cols, default=few, key='f_dims',
                              help='Add or remove filter fields.')
        for col in dims:
            options = analytics.filter_options(data[col])
            selections[col] = st.multiselect(col, options, default=options, key=f'f_{col}')

    filtered = analytics.filter_frame(data, selections)
    st.caption(f'{len(filtered):,} of {len(data):,} rows match the filters.')
    if filtered.empty:
        st.warning('No records match the selected filters. Widen the filters in the sidebar.')
        return

    # ---------- KPIs
    total = analytics.get_total(filtered, measure)
    monthly = analytics.monthly_revenue(filtered, date_col, measure) if date_col else None
    group_options = cat_cols
    # default breakdown: a column with no missing values and a readable number of groups
    complete = [c for c in group_options if not data[c].isna().any() and 3 <= data[c].nunique() <= 12
                and not re.search(r'(?i)year|month', c)]
    default_group = complete[0] if complete else (group_options[0] if group_options else None)
    group_col = (st.selectbox('Break down revenue by', group_options, key='dash_group',
                              index=group_options.index(default_group)) if group_options else None)
    share = analytics.revenue_share(filtered, measure, group_col) if group_col else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f'Total {measure}', money(total.value))
    k2.metric('Records', f'{len(filtered):,}')
    if monthly and monthly.value:
        k3.metric('Average per month', money(sum(monthly.value.values()) / len(monthly.value)))
    else:
        k3.metric('Average per month', 'n/a')
    if share and share.value:
        top_name, top_vals = next(iter(share.value.items()))
        k4.metric(f'Top {group_col}', top_name, f"{top_vals['share_pct']}% of revenue", delta_color='off')
    else:
        k4.metric('Top group', 'n/a')

    # ---------- charts
    left, right = st.columns(2)
    with left:
        st.subheader('Monthly revenue')
        if monthly and monthly.value:
            st.line_chart(pd.Series(monthly.value, name=measure))
            st.caption(' '.join(monthly.caveats))
        else:
            st.info('Choose a date column in the Model tab to see a monthly trend.')
    with right:
        st.subheader('Revenue by year')
        yoy = analytics.yoy_growth(filtered, date_col, measure) if date_col else None
        if yoy and yoy.value:
            table = pd.DataFrame(yoy.value).T
            st.bar_chart(table['total'].rename(measure))
            show = table.rename(columns={'total': measure, 'yoy_pct': 'Change vs prior year (%)',
                                         'months_with_data': 'Months of data'})
            st.dataframe(show)
            st.caption(' '.join(yoy.caveats))
        else:
            st.info('Choose a date column in the Model tab to see yearly revenue.')

    st.subheader(f'Revenue by {group_col}' if group_col else 'Revenue breakdown')
    if share and share.value:
        n_show = st.slider('Show top N groups', 3, 20, 10, key='dash_topn') if len(share.value) > 3 else len(share.value)
        top = analytics.top_n(filtered, measure, group_col, n=n_show)
        b1, b2 = st.columns(2)
        b1.bar_chart(pd.Series(top.value, name=measure))
        table = pd.DataFrame(share.value).T.head(n_show)
        b2.dataframe(table.rename(columns={'total': measure, 'share_pct': 'Share (%)'}))
        st.caption(' '.join(share.caveats))
    else:
        st.info('No category columns are available for a breakdown.')

    # ---------- validation helper
    with st.expander('Validation helper: numbers to compare with Power BI or Excel'):
        active = {k: v for k, v in selections.items() if len(v) < len(analytics.filter_options(data[k]))}
        st.markdown(f'**Filters applied:** {active if active else "none"}')
        st.markdown(f'**Rows:** {len(filtered):,}   **Total {measure}:** {total.value:,.2f}')
        if date_col and yoy and yoy.value:
            st.dataframe(pd.DataFrame(yoy.value).T[['total']].rename(columns={'total': measure}))
    with st.expander('Data quality of the model table'):
        profile = analytics.profile_table(data)
        q1, q2, q3 = st.columns(3)
        q1.metric('Rows', f"{profile['rows']:,}")
        q2.metric('Columns', profile['columns'])
        q3.metric('Duplicate rows', profile['duplicate_rows'])
        st.dataframe(profile['missing'], hide_index=True)
    with st.expander('Filtered records'):
        st.dataframe(filtered, hide_index=True)


# ---------------------------------------------------------------- 5. tabs
tab_data, tab_model, tab_dash, tab_assistant = st.tabs(['Data and profile', 'Model', 'Dashboard', 'Assistant'])

with tab_data:
    st.subheader('Files and tables found')
    st.dataframe(loading.describe_tables(list(loaded.values())), hide_index=True)
    names = list(usable)
    chosen = st.selectbox('Table to profile', names,
                          index=names.index(max(names, key=lambda n: len(usable[n].df))), key='profile_table')
    df = usable[chosen].df
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

model = None
with tab_model:
    model = render_model_tab(usable)

with tab_dash:
    if model is None:
        st.info('Finish the Model tab first.')
    else:
        render_dashboard(model)

with tab_assistant:
    st.caption('Mode: rules-based starter. Describe what you actually built here.')
    if model is None:
        st.info('Finish the Model tab first. The assistant answers from the model table.')
    else:
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
                answer = agent.answer_question(question, model['df'], mode='rules',
                                               value_col=model['measure'], date_col=model['date_col'])
                st.session_state['history'].append({'question': question, 'answer': answer})
                st.rerun()