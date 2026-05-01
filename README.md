## chat_app/chat_frontend.py:
1. Provides chat interface for interacting with LLMs.
2. CSV file transformation via natural language.
3. Persists chat history in different sessions.
4. Uses `using_langgraph.py` for CSV file transformation and `using_langchain.py` for normal interactions.

## chat_app/using_langgraph.py:
This script demonstrates a LangGraph workflow that classifies user input to determine whether to perform an Excel-style data transformation (using pandas) or a simple conversational Q&A.<br/>

  _Methods:_
  - `node_0`: Initializes the conversation by wrapping the raw user prompt in a ChatPromptTemplate.
  - `node_1`: Uses a structured Gemini model to classify the intent of the prompt as either 'excel_transformation' or 'not_excel_transformation'.
  - `node_2`: Activated for Excel transformations; prompts the model to generate pandas-based python transformation code for a dataframe and executes the code 
  - `node_3`: Activated for general queries; uses a standard conversational model to answer the user.
  - `conditional_node`: Routes the workflow to `node_2` or `node_3` based on the classification result from `node_1`.
  <br/>
  
  _LangGraph Nodes:_
  - `node_0`: Prepares the initial system/human prompt structure.
  - `node_1`: Acts as an intelligent router/classifier.
  - `node_2`: Generates code for data manipulation/transformations.
  - `node_3`: Handles standard chat responses.

## chat_app/using_langchain.py:
This script chains prompt template, chat models and string output parser together to take input from end user and provide output produced by LLM.

## chat_models_snippet/*:
The collection of script in this folder contains working with simple, complex chains and interacting with open source and gemini chat models.

## rag/rag_using_langchain_sklearn.py:
This script contains code for Retrieval Augmented Generation (RAG). Created via sklearn, LangChain, gemini embedding model and gemini LLM.

## runnable_snippet/*:
Contains collection of scripts on how to work with runnables of LangChain.
