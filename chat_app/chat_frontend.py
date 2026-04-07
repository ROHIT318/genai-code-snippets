import streamlit as st

with st.container(border=True):

    with st.chat_message('user'):
        st.write('Hello, How are you?')

    with st.chat_message('assistant'):
        st.write('I am good, how are you?')

    st.chat_input('Enter your messages here.')