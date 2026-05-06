from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableBranch, RunnablePassthrough, RunnableLambda, chain
from dotenv import load_dotenv
import os

load_dotenv()

chat_gemini_api_key = os.getenv('chat_gemini_api_key')
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=chat_gemini_api_key)

str_output_parser = StrOutputParser()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Using RunnableParallel

# chat_template_1 = ChatPromptTemplate([
#     ('system', 'You are a helpful AI assistant, having over {yoe_1} years of experience in {role_2} field.'),
#     ('human', '{query_1}')
# ])

# chat_template_2 = ChatPromptTemplate([
#     ('system', 'You are a helpful AI assistant, having over {yoe_2} years of experience in {role_2} field.'),
#     ('human', '{query_2}')
# ])

# parallel_chain = RunnableParallel({
#     'computer_scientist': RunnableSequence(chat_template_1, chat_model, str_output_parser),
#     'data_scientist': RunnableSequence(chat_template_2, chat_model, str_output_parser)
# })

# parallel_chain.get_graph().print_ascii()

# output = parallel_chain.invoke({'yoe_1': 10, 'role_1': 'Data Scientist', 'query_1': 'Explain Gradient Descent in simple terms.',
#                        'yoe_2': 10, 'role_2': 'Data Scientist', 'query_2': 'Explain Gradient Descent in simple terms.'})
# print(output)

# # -----------------------------------------------------------------------------------------------------------------------------------------------
#  Using RunnableBranch
chat_template_3 = ChatPromptTemplate([
    ('system', 'You are a helpful AI assistant, having over 10 years of experience in computer science field. Answer the quesries asked by user.'),
    ('human', 'Answer what is gradient decent under 100 words.')
])

chat_template_4 = ChatPromptTemplate([
    ('system', 'You are a helpful AI assistant, having over 10 years of experience in car mechanic field. Answer the queries asked by user.'),
    ('human', 'Answer what is car battery under 100 words.')
])

# conditional_chain = RunnableBranch(
#     (lambda x: x.get('role')=='Computer Scientist', RunnableSequence(chat_template_3, chat_model, str_output_parser)),
#     (lambda x: x.get('role')=='Car Mechanic', RunnableSequence(chat_template_4, chat_model, str_output_parser)),
#     RunnablePassthrough(lambda content: "Issue with condition.")
# )

# # res = conditional_chain.invoke({'role': 'Computer Scientist'})
# # res = conditional_chain.invoke({'role': 'Car Mechanic'})
# res = conditional_chain.invoke({'role': 'Other Role'})
# print(res)

# # -----------------------------------------------------------------------------------------------------------------------------------------------
# Using custom function for branching

@chain
def RunnableBranchCustom(input_data):
    role = input_data.get('role')
    if role=='Computer Scientist':
        return RunnableSequence(chat_template_3, chat_model, str_output_parser)
    elif role=='Car Mechanic':
        return RunnableSequence(chat_template_4, chat_model, str_output_parser)
    else:
        return 'Issue with condition.'

# runnable_branch_custom = RunnableLambda(RunnableBranchCustom)

# output = RunnableBranchCustom.invoke({'role': 'Computer Scientist'})
# output = RunnableBranchCustom.invoke({'role': 'Car Mechanic'})
output = RunnableBranchCustom.invoke({'role': 'Some Other Role'})
print(output)