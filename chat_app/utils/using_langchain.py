from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv

load_dotenv('../.env')
chat_gemini_api_key = os.getenv('chat_gemini_api_key')

chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=chat_gemini_api_key)

str_output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ('system', '{message_history}'),
    ('system', 'You are a helpful chat assistant help user with their queries.'),
    ('human', 'Query: {query}')
])

sequential_chain = RunnableSequence(prompt, chat_model, str_output_parser)

if __name__ == '__main__':
    message_history =[]
    res = sequential_chain.invoke({'message_history': message_history,'query': 'How are you?'})
    print(res)
