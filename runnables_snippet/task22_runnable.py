from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
import os
from dotenv import load_dotenv

load_dotenv()

chat_gemini_api_key = os.getenv('chat_gemini_api_key')
chat_gemini = ChatGoogleGenerativeAI(model='gemma-3-27b-it', api_key=chat_gemini_api_key)
embedding_gemini_model = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-2-preview', 
    api_key=chat_gemini_api_key, 
    task_type="retrieval_document"
)

input_prompt = """
Task 2.2 — Parallel evidence gathering
Given a business KPI question, build a RunnableParallel that fires three sub-chains concurrently: one retrieves relevant operational context from a FAISS vector store, one searches for the metric definition using a mock tool, and one runs a simpler "quick answer" LLM pass. A final synthesis chain takes all three outputs and produces a single structured response. Measure and log the time saved vs running them sequentially.
"""

# 1. Read the documents: ppt, pdf, txt - DONE
# 2. Chunk the content of documents - DONE
# 3. store them in vectore store - DONE

data_files_dir = []
cwd = os.getcwd()
data_files_dir = 'files'
for _, _, i in os.walk(data_files_dir):
    data_files_dir = list(map(lambda x: os.path.join(cwd, data_files_dir, x), i))

# print(data_files_dir)
txt_files = [file_path for file_path in data_files_dir if 'txt' in file_path ]
# print(txt_files)
ppt_files = [file_path for file_path in data_files_dir if 'ppt' in file_path ]
pdf_files = [file_path for file_path in data_files_dir if 'pdf' in file_path ]

## Read unstructured data and create Documents
txt_document = []
for file_path in txt_files:
    txt_loader = TextLoader(file_path)
    txt_document.extend(txt_loader.load())
# print(txt_document)
print('Length of txt document: ', len(txt_document))

pdf_document = []
for file_path in pdf_files:
    pdf_loader = PyPDFLoader(file_path)
    pdf_document.extend(pdf_loader.load())
# print(pdf_document)
print('Length of pdf document: ', len(pdf_document))

ppt_document = []
for file_path in ppt_files:
    ppt_loader = UnstructuredPowerPointLoader(file_path)
    ppt_document.extend(ppt_loader.load())
# print(ppt_document)
print('Length of ppt document: ', len(txt_document))

## Split the texts into multiple chunks based on charcater size and overlap size
rec_char_txt_splitter_obj = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=16, separators=['\n\n', '\n', ' '])
split_txt_documents = rec_char_txt_splitter_obj.split_documents(txt_document)
print(f'Length of txt_documents: {len(txt_document)}, length of split txt docuents: {len(split_txt_documents)}')

split_pdf_documents = rec_char_txt_splitter_obj.split_documents(pdf_document)
print(f'Length of txt_documents: {len(pdf_document)}, length of split txt docuents: {len(split_pdf_documents)}')

split_ppt_documents = rec_char_txt_splitter_obj.split_documents(ppt_document)
print(f'Length of txt_documents: {len(ppt_document)}, length of split txt docuents: {len(split_ppt_documents)}')

# for document in split_pdf_documents:
#     print(document.page_content, end='\n\n')

vectorstore = FAISS.from_documents(
    documents=split_txt_documents + split_pdf_documents + split_ppt_documents,
    embedding=embedding_gemini_model
)

retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k':5, 'lambda_mult':1})
res = retriever.invoke('What is gradient descent?')
print(res)