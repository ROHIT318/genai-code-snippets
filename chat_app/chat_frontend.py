import streamlit as st
from using_langchain import sequential_chain

# variable to store our messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'user', 'message': 'Hello, How are you?'},
                     {'role': 'assistant', 'message': 'I am good, how are you?'}]

with st.container(border=True):

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['message'])

    col_1, col_2 = st.columns(2, [1, 10])
    
    with col_1: 
        files = st.file_uploader()

    with col_2:
        if user_message := st.chat_input('Enter your messages here.'):
            st.session_state['messages'].append({'role': 'user', 'message': user_message})
            ai_message = sequential_chain.invoke({'message_history': st.session_state['messages'], 'query': user_message})
            st.session_state['messages'].append({'role': 'assistant', 'message': ai_message})
            st.rerun()