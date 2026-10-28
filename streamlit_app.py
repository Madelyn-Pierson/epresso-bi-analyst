"""Main app. Checkpoint 0: loads a file and shows a profile.
Keep calculations in analytics.py and the assistant in agent.py."""
import streamlit as st

import agent
import analytics

MAX_QUESTIONS = 15  # per-session limit for the assistant

st.set_page_config(page_title='E-Presso BI Analyst', layout='wide')
st.title('E-Presso BI Analyst')
st.caption('Starter app. Replace this line with a description of your own application.')


@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, filename: str, sheet_name):
    # The cache is keyed on the file CONTENTS, so a modified file is never
    # confused with the original one.
    return analytics.load_table(file_bytes, filename, sheet_name)


with st.sidebar:
    st.header('Data')
    uploaded = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'txt', 'tsv', 'xlsx', 'xls'])
    sheet = 0
    if uploaded is not None and uploaded.name.lower().endswith(analytics.EXCEL_EXT):
        sheet = st.selectbox('Sheet', analytics.excel_sheet_names(uploaded.getvalue()))

if uploaded is None:
    st.info('Upload a file in the sidebar to begin.')
    st.stop()

try:
    df = cached_load(uploaded.getvalue(), uploaded.name, sheet)
except Exception as exc:  # show a readable message instead of a traceback
    st.error(f'Could not read that file: {exc}')
    st.stop()

tab_data, tab_dash, tab_assistant = st.tabs(['Data and profile', 'Dashboard', 'Assistant'])

with tab_data:
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
