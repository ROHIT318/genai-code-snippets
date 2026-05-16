import streamlit as st
import pandas as pd
import tempfile
import uuid
import json
import os
import re
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from utils.using_langchain import sequential_chain
from utils.using_langgraph import final_workflow, StateSchema, ATSScoreSchema
from utils.excel_transformation import ExcelTransformation
from utils.content_loader import ContentLoader

load_dotenv("../.env")
FILE_TO_USE = os.getenv("FILE_TO_USE")

if "res_df" not in st.session_state:
    st.session_state.res_df = pd.DataFrame()

if "json_res_df" not in st.session_state:
    st.session_state.json_res_df = {}

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "json_pdf_data" not in st.session_state:
    st.session_state.json_pdf_data = []

if "res_pdf_data" not in st.session_state:
    st.session_state.res_pdf_data = []

if "json_csv_data" not in st.session_state:
    st.session_state.json_csv_data = []

if "json_output_pdf" not in st.session_state:
    st.session_state.json_output_pdf = []

# --- CONFIGURATION & HELPERS ---
CHAT_DIR = Path("chat")
CHAT_DIR.mkdir(exist_ok=True)

def get_session_files():
    """Retrieve all chat session files sorted by creation time."""
    files = list(CHAT_DIR.glob("*.json"))
    return sorted(files, key=os.path.getmtime, reverse=True)

def save_message(session_id, role, content, json_csv_data=None, json_pdf_data=None):
    """Append a message to the session"s JSON file."""
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
    
    msg_entry["json_csv_data"] = json_csv_data
    msg_entry["json_pdf_data"] = json_pdf_data

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

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- SIDEBAR: SESSION MANAGEMENT ---
with st.sidebar:
    st.header("Chat Sessions")
    if st.button("➕ New Chat"):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.rerun()
    
    st.divider()
    for file in get_session_files():
        with open(file, "r") as f:
            try:
                data = json.load(f)
                title = data.get("title") or ([m["content"] for m in data["messages"] if m["role"] == "user"] or ["Empty Chat"])[0][:20]
                
                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                if col1.button(title, key=f"btn_{file.name}"):
                    st.session_state["session_id"] = data["session_id"]
                    st.session_state["messages"] = data["messages"]
                    st.rerun()
                if col2.button("", icon="✏️", key=f"edit_{file.name}"):
                    st.session_state["renaming"] = data["session_id"]
                if col3.button("", icon="🗑️", key=f"delete_{file.name}"):
                    st.session_state["deleting"] = data["session_id"]
            except (json.JSONDecodeError, KeyError):
                continue
    
    if "renaming" in st.session_state:
        new_title = st.text_input("Rename chat:", key="rename_input")
        if st.button("Save"):
            update_session_title(st.session_state["renaming"], new_title)
            del st.session_state["renaming"]
            st.rerun()

    if "deleting" in st.session_state:
        file_path = CHAT_DIR / f"{st.session_state["deleting"]}.json"
        if file_path.exists():
            os.remove(file_path)
            del st.session_state["deleting"]
            st.rerun()

# --- MAIN WINDOW ---
st.title("💬 AI Chat Assistant")
# Display Chat History
for message in st.session_state.messages:
    role = "assistant" if message["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.write(message["content"])
        try:
            if "json_csv_data" in message and message["json_csv_data"] is not None and len(message["json_csv_data"])!=0:
                st.session_state.df = json.loads(message["json_csv_data"])
                st.dataframe(pd.DataFrame(st.session_state.df))
            elif "json_pdf_data" in message and message["json_pdf_data"] is not None and len(message["json_pdf_data"])!=0:
                # st.write(message["json_pdf_data"][2][1])
                for key, value in message["json_pdf_data"][0].items():
                    st.write(f"""
                            **For JOB Title: {key}**. \n
                            **your ATS score is: {message["json_pdf_data"][1][key]}**.\n
                            **Below improvements can be implemented to improve your score:**\n
                            """)
                    pattern = r'(\d+\..*?)(?=\d+\.|$)'
                    points = re.findall(pattern, str({message["json_pdf_data"][2][key]})[2:-2], re.DOTALL)
                    for point in points:
                        st.info(point)
                    st.divider()
        except:
            # st.write(message["json_pdf_data"][2][1])
            continue

# Input handling
if prompt := st.chat_input("Enter your messages here", accept_file=True, file_type=["csv", "pdf"]):
    user_message = prompt.text if hasattr(prompt, "text") else prompt
    json_csv_data = None
    temp_path = ""
    
    # Handle CSV upload
    if len(prompt.files) != 0:
        file_name = prompt.files[0].name.lower()

        try:

            if file_name.endswith(".csv"):
                df = pd.read_csv(prompt.files[0])
                # Input tabular content provided by user.
                st.session_state.json_csv_data = df.to_json()

            elif file_name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(dir=os.path.join(os.getcwd(), "chat_app", "utils"), delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(prompt.files[0].getvalue())
                    temp_path = temp_file.name

                content_loader = ContentLoader([temp_path,])
                # st.write(content_loader.display_all_documents())
                # Input pdf content provided by user.
                st.session_state.json_pdf_data = content_loader.clean_document_content
                
        except Exception as e:
            st.error("Error reading file, {e}.")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Uer input is saved in session_state to be used for display
    st.session_state["messages"].append({"role": "user", "content": user_message, "json_csv_data": st.session_state.json_csv_data, "json_pdf_data": st.session_state.json_pdf_data})
    # Save the output into chat history database
    save_message(st.session_state["session_id"], "user", user_message, json_csv_data=st.session_state.json_csv_data, json_pdf_data=st.session_state.json_pdf_data)
    
    with st.chat_message("user"):
        
        # Display tabular data if end user provides it otherwise only string content
        st.write(user_message)
        if st.session_state.json_csv_data:
            st.dataframe(json.loads(st.session_state.json_csv_data))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if FILE_TO_USE=="using_langchain.py":

                ai_response = sequential_chain.invoke({
                    "message_history": st.session_state["messages"], 
                    "query": user_message
                })["text"]
                st.write(ai_response)
                save_message(session_id=st.session_state["session_id"], role="assistant", content=ai_response)

            else:

                # If pdf was provided by end user
                if st.session_state.json_csv_data or not st.session_state.json_pdf_data:
                    ai_response = final_workflow.invoke({
                        "message_history": st.session_state["messages"], 
                        "prompt": user_message + "Columns present in input dataset: " + pd.DataFrame(st.session_state.df).columns
                    })
                    query_type = ai_response["query_type"]
                    ai_response = ai_response["output"]

                    # If tabular data transformation is required
                    if query_type=="excel_transformation" and not pd.DataFrame(st.session_state.df).empty:
                        xl_trsnf = ExcelTransformation(df=pd.DataFrame(st.session_state.df), steps=ai_response)
                        c = xl_trsnf.df
                        st.session_state.json_res_df = xl_trsnf.df.to_json()
                    st.write(ai_response)
                    save_message(session_id=st.session_state["session_id"], role="assistant", content=ai_response, json_csv_data=st.session_state.json_res_df)
                    st.session_state["messages"].append({"role": "assistant", "content": ai_response, "json_csv_data": st.session_state.json_res_df, "json_pdf_data": st.session_state.json_output_pdf})

                else:

                    input = StateSchema(
                        prompt=user_message, 
                        resume=st.session_state.json_pdf_data,
                    )
                    ai_response = ATSScoreSchema()
                    ai_response = final_workflow.invoke({
                        "prompt": user_message,
                        "resume_details": st.session_state.json_pdf_data
                    })
                    ai_response = final_workflow.invoke(input)

                    st.session_state.json_output_pdf.append(ai_response["job_description"])
                    st.session_state.json_output_pdf.append(ai_response["ats_score"])
                    st.session_state.json_output_pdf.append(ai_response["job_wise_improvements"])
                    save_message(session_id=st.session_state["session_id"], role="assistant", content="Below is the result of analysis: ", json_pdf_data=st.session_state.json_output_pdf)
                    st.session_state["messages"].append({"role": "assistant", "content": "Below is the result of analysis:", "json_csv_data": st.session_state.json_csv_data, "json_pdf_data": st.session_state.json_output_pdf})


                # elif st.session_state.json_pdf_data is not None and len(st.session_state.json_pdf_data)!=0:
                #     ai_response = final_workflow.invoke({
                #         "message_history": st.session_state["messages"], 
                #         "prompt": user_message,
                #         "resume_details": st.session_state.json_pdf_data
                #     })

                # for job, value in st.session_state.res_pdf_data["ats_score"].items():
                #     st.write(f"Job Title is {job}. ATS Score for the job is {value}")

            # if not st.session_state.res_df.empty:
            #     st.dataframe(st.session_state.res_df)
            # elif st.session_state.res_pdf_data!=None and len(st.session_state.res_pdf_data)!=0:
            #     st.write(st.session_state.res_pdf_data)
    st.session_state.json_csv_data = []
    st.session_state.json_pdf_data = []
    st.session_state.json_output_pdf = []
    st.session_state.res_df = pd.DataFrame()
    st.rerun()
