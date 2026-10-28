"""Main app. Stage 1: upload ONE OR MORE files (CSV / TXT / Excel / Access / SQLite),
pick the table to analyze, and see a profile of it. Nothing here assumes a particular dataset.
Keep calculations in analytics.py, file reading in loading.py, and the assistant in agent.py."""
import streamlit as st

import agent
import analytics
import loading

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
    st.info('TODO: KPIs, filters, and at least three charts go here. '
            'Calculate everything from the loaded data. Never hardcode results.')

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