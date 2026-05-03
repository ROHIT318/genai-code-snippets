from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Annotated, Literal, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv('../.env')
chat_gemini_api_key = os.getenv('chat_gemini_api_key')
str_output_parser = StrOutputParser()

# 0. node_0 is a chat prompt template which prepares the prompt to be sent to chat_model
# 1. Will classify in the chat itself whether it is an excel transformation operation or a simple query based task.
    # 1.1. To be done by node 1 and chat_model_1.
    # 1.2. It needs to produce structured output.
# 2. If it is an excel based transformation then redirect to node_2.
    # 2.1. chat_model_2 will produce a list of steps. 
    # 2.2. execute those list of steps one after another.
# 3. If it is not an excel based transformation then redirect to node_3
    # 3.1. Simple Q&A will continue.


class ChatModelSchema(BaseModel):
    """ Output schema to be produced by chat_model_1 """
    query_type: Annotated[str, Literal['excel_transformation', 'not_excel_transformation']] = Field(description='Output can either be "excel_transformation" or "not_excel_transformation" and nothing else allowed.') 
    output: List[str] = Field(description='In case it is asking for "excel_transformation" then return blank else return the query as is.')


# chat_model_1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=chat_gemini_api_key)
chat_model_1 = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview', api_key=chat_gemini_api_key)
# Converting chat model so that output structure is of type ChatModelOutput
structured_chat_model_1 = chat_model_1.with_structured_output(ChatModelSchema)


# state variable for the graph
class StateSchema(TypedDict):
    prompt: str
    message_history: List[Dict[str, str]]
    # query_type: Annotated[str, Literal['excel_transformation', 'not_excel_transformation']] = Field(description='Output is produced will on the basis of prompt provided and can either be excel_transformation and not_excel_transformation.') 
    # Not doing strict type checking it is already handled by ChatModelSchema 
    query_type: str
    output: List[str]


def node_0(input_state: StateSchema):
    query = input_state['prompt']
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful AI assistant. Answer the queries from user." '),
        ('human', '{query}')
    ])
    input_state['prompt'] = prompt_template.invoke({'query': query})
    print('Executed Node 0 !!')
    return input_state


def node_1(input_state: StateSchema):
    res = structured_chat_model_1.invoke(input_state['prompt'])
    input_state['query_type'] = res.query_type
    print('Executed Node 1 !!')
    return input_state


# chat_model_2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=chat_gemini_api_key)
chat_model_2 = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview', api_key=chat_gemini_api_key)
def node_2(input_state: StateSchema):
    query = input_state['prompt']
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful AI assistant that provides back the python code for transformation of a pandas dataframe based on user query. Provide only the python code and nothing else. Each python code line will be seaparated by a comma.'),
        ('human', '{query}')
    ])
    # Want to store my final prompt
    input_state['prompt'] = prompt_template.invoke(query)
    input_state['output'] = (chat_model_2 | str_output_parser).invoke(input_state['prompt'])
    print('Executed Node 2 !!')
    return input_state


def node_3(input_state: StateSchema):
    query = input_state['prompt']
    prompt_template = ChatPromptTemplate([
        ('system', 'You are a helpful AI chat assistant. Answer the queries of user to the best of your knowledge.'),
        ('human', '{query}')
    ])
    # Want to store my final prompt
    input_state['prompt'] = prompt_template.invoke(query)
    input_state['output'] = (chat_model_2 | str_output_parser).invoke(query)
    print('Executed Node 3 !!')
    return input_state


def conditional_node(input_state: StateSchema):
    if input_state['query_type']=='excel_transformation':
        return "node_2"
    else:
        return "node_3"

graph = StateGraph(StateSchema)

graph.add_node('node_0', node_0)
graph.add_node('node_1', node_1)
graph.add_node('node_2', node_2)
graph.add_node('node_3', node_3)

graph.add_edge(START, 'node_0')
graph.add_edge('node_0', 'node_1')
graph.add_conditional_edges('node_1', conditional_node, {
        "node_2": "node_2",
        "node_3": "node_3",   
    }
)
graph.add_edge('node_2', END)
graph.add_edge('node_3', END)

final_workflow = graph.compile()

if __name__ == '__main__':
    # input_state = {'prompt': 'How are you?'}
    input_state = {"prompt": "Add the column 'a' and 'b' in table tab to produce a final table having column 'res'. Further on, create another column called 'mul_res' which is produced by a*b. Create another column 'even_or_odd': if a is even then the new column stores 'Even' else 'Odd'."}
    res = final_workflow.invoke(input_state)
    print(f'Prompt is {res}')