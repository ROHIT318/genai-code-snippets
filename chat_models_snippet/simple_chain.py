from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

chat_gemini_api_key = os.getenv('chat_gemini_api_key')

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful AI assistant, having over {yoe} years of experience in {role} field.'),
    ('human', '{query}')
])

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=chat_gemini_api_key)
# chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=chat_gemini_api_key)

str_output_parser = StrOutputParser()

chain = chat_template | chat_model | str_output_parser

output = chain.invoke({'yoe': 10, 'role': 'Data Scientist', 'query': 'Explain Gradient Descent in simple terms.'})
print(output)