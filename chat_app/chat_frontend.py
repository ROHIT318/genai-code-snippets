import streamlit as st

# variable to store our messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'user', 'message': 'Hello, How are you?'},
                     {'role': 'assistant', 'message': 'I am good, how are you?'}]

with st.container(border=True):

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['message'])

    if user_message := st.chat_input('Enter your messages here.'):
        st.session_state['messages'].append({'role': 'user', 'message': user_message})
        st.rerun()