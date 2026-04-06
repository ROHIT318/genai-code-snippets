from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Annotated

# Initialize Ollama model
llm = OllamaLLM(model="hermes3")

chat_prompt = ChatPromptTemplate.from_messages([
    ('human', 'You are a helpful AI assistant, having over {yoe} years of experience in {role} field.'),
    ('system', 'Noted, I will adhere to the requests and here is step by step procedure.'),
    ('human', '{user_prompt}')
])

# # Prompt Creation
# yoe = input('What is the years of experience for AI? ')
# role = input('In which field is the experience? ')
# user_prompt = input('Ask your query? ')

yoe = 3
role = 'Car Mechanic'
user_prompt = 'How to create car batteries, easily?'

formatted_prompt = chat_prompt.invoke({'yoe': yoe, 'role': role, 'user_prompt': user_prompt})
print(formatted_prompt)


# Send prompt
# response = llm.invoke("How are you?", stream=True)
response = llm.invoke(formatted_prompt, stream=True)    
print(response)