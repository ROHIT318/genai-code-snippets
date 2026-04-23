import streamlit as st
import pandas as pd
import uuid
import json
import os
from datetime import datetime
from pathlib import Path
from using_langchain import sequential_chain

# --- CONFIGURATION & HELPERS ---
CHAT_DIR = Path("chat")
CHAT_DIR.mkdir(exist_ok=True)

def get_session_files():
    """Retrieve all chat session files sorted by creation time."""
    files = list(CHAT_DIR.glob("*.json"))
    return sorted(files, key=os.path.getmtime, reverse=True)

def save_message(session_id, role, content, csv_data=None):
    """Append a message to the session's JSON file."""
    file_path = CHAT_DIR / f"{session_id}.json"
    
    if file_path.exists():
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"session_id": session_id, "created_at": datetime.now().isoformat(), "messages": [], "title": None}
    else:
        data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "title": None
        }
    
    msg_entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    if csv_data is not None:
        msg_entry["csv_data"] = csv_data
    
    data["messages"].append(msg_entry)
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def update_session_title(session_id, new_title):
    file_path = CHAT_DIR / f"{session_id}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
        data["title"] = new_title
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

# --- APP SETUP ---
st.set_page_config(page_title="AI Chat Assistant", page_icon="💬", layout="wide")

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# --- SIDEBAR: SESSION MANAGEMENT ---
with st.sidebar:
    st.header("Chat Sessions")
    if st.button("➕ New Chat"):
        st.session_state['session_id'] = str(uuid.uuid4())
        st.session_state['messages'] = []
        st.rerun()
    
    st.divider()
    for file in get_session_files():
        with open(file, "r") as f:
            try:
                data = json.load(f)
                title = data.get("title") or ([m['content'] for m in data['messages'] if m['role'] == 'user'] or ["Empty Chat"])[0][:20]
                
                col1, col2 = st.columns([0.8, 0.2])
                if col1.button(title, key=f"btn_{file.name}"):
                    st.session_state['session_id'] = data['session_id']
                    st.session_state['messages'] = data['messages']
                    st.rerun()
                if col2.button("✏️", key=f"edit_{file.name}"):
                    st.session_state['renaming'] = data['session_id']
            except (json.JSONDecodeError, KeyError):
                continue
    
    if 'renaming' in st.session_state:
        new_title = st.text_input("Rename chat:", key="rename_input")
        if st.button("Save"):
            update_session_title(st.session_state['renaming'], new_title)
            del st.session_state['renaming']
            st.rerun()

# --- MAIN WINDOW ---
st.title("💬 AI Chat Assistant")

# Display Chat History
for message in st.session_state.messages:
    role = "assistant" if message['role'] == 'assistant' else "user"
    with st.chat_message(role):
        st.write(message['content'])
        if "csv_data" in message:
            df = pd.read_json(message["csv_data"])
            st.dataframe(df)

# Input handling
if prompt := st.chat_input('Enter your messages here', accept_file=True, file_type=["csv"]):
    user_message = prompt.text if hasattr(prompt, 'text') else prompt
    csv_data = None
    
    # Handle CSV upload
    if hasattr(prompt, 'file') and prompt.file is not None:
        try:
            df = pd.read_csv(prompt.file)
            csv_data = df.to_json()
        except Exception as e:
            st.error("Error reading CSV file.")
    
    st.session_state['messages'].append({'role': 'user', 'content': user_message, 'csv_data': csv_data})
    save_message(st.session_state['session_id'], "user", user_message, csv_data=csv_data)
    
    with st.chat_message("user"):
        st.write(user_message)
        if csv_data:
            st.dataframe(pd.read_json(csv_data))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_response = sequential_chain.invoke({
                'message_history': st.session_state['messages'], 
                'query': user_message
            })
            st.write(ai_response)
    
    st.session_state['messages'].append({'role': 'assistant', 'content': ai_response})
    save_message(st.session_state['session_id'], "assistant", ai_response)
    st.rerun()
