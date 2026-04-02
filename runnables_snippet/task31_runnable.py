# Module 3 — Production Hardening
# Task 3.1 — Fallback chain with model failover
# Build a primary chain using a gpt-4o-mini (or Claude Haiku equivalent) call. Wrap it with .with_fallbacks([backup_chain]) where the backup uses a different provider. Simulate the primary failing by injecting a RunnableLambda that raises an exception 50% of the time (use random). Log which chain actually served each request. Run 20 invocations and report success/fallback ratio.

# 1. make a chat model with gpt-4o-mini, gemma, haiku - done
# 2. make the chain fault tolerant: gemma, haiku; use with_fallbacks, use only in case of ratelimiterror - done
# 3. Inside the chain itself introduce RunnableLambda to raise error 50% of the time - done
# 4. Log request details, (model name, query, output, status) 
# 5. 20 invocations and report sucess/fallback ratio 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain_core.exceptions import RateLimitError
from langchain_openai import ChatOpenAI
from langchain_anothropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import random
import os
from dotenv import load_dotenv

load_dotenv()
api_key_1 = os.getenv('gpt_chat_model')
api_key_2 = os.getenv('gemma_chat_model')
api_key_3 = os.getenv('haiku_chat_model')

gpt_chat_model = ChatOpenAI(model='', api_key='')
gemma_chat_model = ChatGoogleGenerativeAI(model='', api_key='')
haiku_chat_model = ChatAnthropic(model='', api_key='')

query = 'You have a persona of {role} and year of experience as {yoe}. Do this {task}.'
prompt_template = ChatPromptTemplate.from_template(query)

str_parser = StrOutputParser()

def handle_error(model, prompt, output, status):
    db.create_entry({'model': model, 'prompt': prompt, 'output': output, 'status': status})
    raise TimeoutError


chain_1 = RunnableSequence(
    RunnablePassthrough.assign(
        final_prompt=RunnableLambda(lambda x: prompt_template.invoke(x)), 
        num=random.random()
    ),
    RunnableLambda(
        lambda x: handle_error('gpt-chat-model', x.get('final_prompt'), '', 'Fall Back') 
        if x.get('num') < 0.5 
        else x.get('final_prompt')
    },
    gpt_chat_model, 
    str_parser 
)

chain_2 = RunnableSequence(prompt_template, gemma_chat_model, str_parser)
chain_3 = RunnableSequence(prompt_template, haiku_chat_model, str_parser)

fault_tolerant_chain = chain_1.with_fallbacks([chain_2, chain_3], exceptions_to_handle=[RateLimitError, TimeoutError])

query_input = {'role': role, 'yoe': yoe, 'task': task}
for i in range(20):
    query['num'] = i
    res = fault_tolerant_chain.invoke(query_input)
    print(res)