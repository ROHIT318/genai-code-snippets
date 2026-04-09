from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('chat_gemini_api_key')

# ---------------------------- Text Loader -------------------------------------
# txt_loader = TextLoader('files/doc.txt')
# text_content = txt_loader.load()
# print(type(text_content)) # List
# print(len(text_content)) # Length of the list.
# print(text_content[0].page_content) # Content of the file
# print(text_content[0].metadata) # {'source': 'files/doc.txt'}
# ---------------------------- Text Loader -------------------------------------

# ---------------------------- PDF Loader -------------------------------------
# pdf_loader = PyPDFLoader('files/gradient_descent.pdf')
# pdf_content = pdf_loader.load()
# print(type(pdf_content)) # List
# print(len(pdf_content)) # Length of the list.
# print(pdf_content[0].page_content) # Content of the file
# print(text_pdf_contentcontent[0].metadata) # {'source': 'files/gradient_descent.pdf'}
# ---------------------------- PDF Loader -------------------------------------


# ---------------------------- PPT Loader -------------------------------------
# # ppt_loader = UnstructuredPowerPointLoader('files/decision_trees.pptx', mode='elements')
# ppt_loader = UnstructuredPowerPointLoader('files/decision_trees.pptx')

# ppt_content = ppt_loader.load()
# print(type(ppt_content)) # List
# print(len(ppt_content)) # Length of the list.
# print(ppt_content[0].page_content) # First document content

# # for i, content in enumerate(ppt_content):
# #     print(i, content)

# print(ppt_content[0].metadata) # {'source': 'files/pd.ppt'}
# ---------------------------- PPT Loader -------------------------------------

class output_structure(BaseModel):
    output: Annotated[Literal['Car Mechanic', 'Gradient Descent', 'Decision Tree'], 
                      Field(description='category of the query asked by user.')]
    
gemini_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=gemini_api_key)
str_gemini_mdoel = gemini_model.with_structured_output(output_structure)

car_query = 'What are the chemicals used in car battery?'
gd_query = 'In gradient descent, finding slope is the most important part?'
dt_query = 'In decision tree top nodes or starting nodes, are called lead nodes?'
nonsense_query = 'Who is linear regression?'

res = str_gemini_mdoel.invoke(gd_query)
print(res)


