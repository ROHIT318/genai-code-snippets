from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from typing import Annotated, Literal
from dotenv import load_dotenv
import os

load_dotenv()
chat_gemini_api_key = os.getenv('chat_gemini_api_key')

class output_schema(BaseModel):
    question: Annotated[str, 'Original query asked by the user.']
    query_scope: Annotated[Literal['forecast', 'bottleneck analysis', 'general', 'others'], 'Category of the query.']

chat_gemini = ChatGoogleGenerativeAI(model='gemma-3-27b-it', api_key=chat_gemini_api_key)
structured_chat_gemini = chat_gemini.with_structured_output(output_schema, strict=True)
str_output_parser = StrOutputParser()

input_prompt = """
    {"question": "Build a chain that classifies an incoming query into one of three intents — forecast, bottleneck_analysis, general — and then routes it to a dedicated sub-chain per intent. Each sub-chain has its own system prompt tuned for that domain. Use RunnableBranch for the routing. The classifier itself must be a lightweight LCEL chain (no hardcoded if/else outside the branch). Handle the case where classification returns an unexpected label gracefully."}
"""

forecast_prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are a helpful forecast analysis ai assistant. Ignore all the queries asked other the forecast analysis.'),
    ('human', 'answer the {question} asked by user with in 100 words.')
])

bottleneck_analysis_prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are a helpful bottleneck analysis ai assistant. Ignore all the queries asked other the bottleneck analysis.'),
    ('human', 'answer the {question} asked by user with in 100 words.')
])

general_prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are a helpful ai assistant. Ignore all the queries asked regarding bottleneck and forecast analysis.'),
    ('human', 'answer the {question} asked by user with in 100 words.')
])

conditional_chain = RunnableBranch(
    (lambda x: x.query_scope=='forecast', forecast_prompt | chat_gemini | str_output_parser),
    (lambda x: x.query_scope=='bottleneck analysis', bottleneck_analysis_prompt | chat_gemini | str_output_parser),
    (lambda x: x.query_scope=='general', general_prompt | chat_gemini | str_output_parser),
    RunnableLambda(lambda x: 'Your query can not be answered.'),
)

input_prompt_forecast = """
    Important forecasting Points.
"""

input_prompt_bottleneck_analysis = """
    Important bottleneck analysis Points.
"""

input_prompt_general = """
    Important general Points.
"""

input_prompt_other = """
    This should fall under others categories.
"""

sequential_parent_chain = RunnableSequence(structured_chat_gemini, conditional_chain)

# res = structured_chat_gemini.invoke(input_prompt_bottleneck_analysis)
# print(res)
# print(res.question)
# print(res.query_scope)

# res = sequential_parent_chain.invoke(input_prompt_general)
# print(res)

res = chat_gemini.invoke(input_prompt_general)
print(res)