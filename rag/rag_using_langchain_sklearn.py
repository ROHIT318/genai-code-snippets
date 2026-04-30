from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
chat_geimini_api_key = os.getenv('chat_gemini_api_key')
emb_model = os.getenv('emb_model')

class RAG():
    def __init__(self, documents: List[str], query: str, emb_model: str, api_key: str, topn: int):
        self.emb_model = GoogleGenerativeAIEmbeddings(model=emb_model, google_api_key=api_key)
        self.documents = documents
        self.documents_embeddings = self.create_embeddings(self.documents)
        self.query = query
        self.query_embeddings = self.embed_query(query)
        self.topn = topn
        self.top_n_similar_documents = self._get_topn_results()

    def create_embeddings(self, documents: List[str]):
        return self.emb_model.embed_documents(self.documents)

    def display_data(self):
        print(f'Documents: {self.documents}\nDocuments Embedding: {self.documents_embedding}\nEmbedding Model: {self.emb_model}')

    def embed_query(self, query: str):
        return self.emb_model.embed_query(query)
    
    def _get_topn_results(self):
        similarity_score = cosine_similarity([self.query_embeddings], self.documents_embeddings)[0]
        document_with_embeddings = list(zip(similarity_score, self.documents))
        document_with_embeddings_sorted = sorted(document_with_embeddings, key=lambda x: x[0], reverse=True)
        top_n_similar_documents = []
        for i in range(self.topn):
            top_n_similar_documents.append(document_with_embeddings_sorted[i][1])
        return top_n_similar_documents
        # return document_with_embeddings

    

if __name__ == '__main__':
    documents = [
        'Delhi is capital of India',
        'Kolkata is capital of West Bengal which is a city in India',
        'Bhubaneshwar is capital of Orissa',
        'Mumbai is capital of Mahrastra',
        'There are 29 states present in India',
        'WBSEDCL is a givernment owned electricity supplying company in West Bengal'
    ]
    query = 'What is the capital of India?'
    rag = RAG(documents=documents, query=query, emb_model=emb_model, api_key=chat_geimini_api_key, topn=2)

    print(rag.top_n_similar_documents)  