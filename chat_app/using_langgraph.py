from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Annotated, Literal
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv('../.env')
chat_gemini_api_key = os.getenv('chat_gemini_api_key')

# 0. node_0 is a chat prompt template which prepares the prompt to be sent to chat_model
# 1. Will classify in the chat itself whether it is an excel transformation operation or a simple query based task.
    # 1.1. To be done by node 1 and chat_model_1.
    # 1.2. It needs to produce structured output.
# 2. If it is an excel based transformation then redirect to node_2.
    # 2.1. chat_model_2 will produce a list of steps. 
    # 2.2. execute those list of steps one after another.
# 3. If it is not an excel based transformation then redirect to node_3
    # 3.1. Simple Q&A will continue.

# Output schema to be produced by chat_model_1
class ChatModelOutput(BaseModel):
    query_type: Annotated[str, Literal['excel_transformation', 'not_excel_transformation'], 'Output is produced will on the basis of prompt provided and can either be excel_transformation and not_excel_transformation.'] 
    output: List[str]

chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=chat_gemini_api_key)
# Converting chat model so that output structure is of type ChatModelOutput
structured_chat_model = chat_model.with_structured_output(ChatModelOutput)

# state variable for the graph
class StateSchema(TypedDict):
    prompt: str
    query_type: Annotated[str, Literal['excel_transformation', 'not_excel_transformation'], 'Output is produced will on the basis of prompt provided and can either be excel_transformation and not_excel_transformation.'] 
    output: List[str]

def node_0(input_state: StateSchema):
    input_state['prompt'] = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpfule AI assistant which will classify whether a user query is of type: "excel_transformation", "not_excel_transformation" '),
        ('human', '{query}')
    ])
    return input_state['prompt']

def node_1(input_state: StateSchema):
    input_state['query_type'] = structured_chat_model.invoke(input_state['prompt'])
    return input_state['query_type'] 

graph = StateGraph(StateSchema)
