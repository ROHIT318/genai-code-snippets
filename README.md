# genai-util-tool

**genai-util-tool** is an intelligent Swiss Army knife for automated career and data workflows. Designed for job seekers navigating the "ATS black hole" and data analysts looking to bypass manual spreadsheet manipulation, this suite leverages the reasoning capabilities of Google Gemini and LangGraph to turn natural language into actionable results. Whether you are optimizing a resume for a specific role or transforming complex datasets with simple English commands, this tool bridges the gap between raw information and high-value output.

## Problem it Solves

* **For Job Seekers:** Most resumes never reach a human recruiter because they lack the specific semantic alignment required by Applicant Tracking Systems (ATS). Manually tailoring a resume for every job application is time-consuming and often involves guesswork.
* **For Data Analysts:** Non-technical users often struggle with complex Excel formulas or Python's Pandas syntax. Performing iterative data cleaning and feature engineering usually requires a developer, creating a bottleneck in business intelligence.

## Features

### 📄 ATS Resume Scorer & Optimizer
Upload your resume in PDF format and provide a job description URL. The tool scrapes the live job posting, performs a semantic gap analysis, and provides:
* An objective ATS compatibility score.
* Actionable, point-by-point improvements to align your skills with the role.
* Certification and skill-set recommendations.

### 📊 Natural Language CSV Transformer
Interact with your data using plain English. Upload a CSV or Excel file and ask the assistant to perform transformations like "Calculate the profit margin," "Filter for rows where sales > 5000," or "Categorize users by age group." The tool generates and executes the necessary code on the fly.

### 🪧 Demo


### 🤖 Intelligent Chat & RAG
A versatile conversational interface that maintains context and utilizes Retrieval-Augmented Generation (RAG) to provide grounded answers based on your uploaded documents.
*[GIF coming soon]*

## How It Works: The LangGraph Pipeline

The core of the data transformation feature is built on a stateful **LangGraph** workflow. This allows for reliable intent classification and specialized execution paths.

```text
[START] --> [Node_0: Prompt Prep]
                 |
        [Node_1: Intent Classifier] 
                 |
        {Conditional Routing}
        /        |           \
 [Node_2]     [Node_3]      [Node_4]
 (Pandas)     (ATS/JD)      (General)
    |            |             |
  [END] <-------[END] <-------[END]
```

## Getting Started
1. Clone the repository
```cmd
git clone https://github.com/ROHIT318/genai-util-tool.git
cd genai-util-tool
```

2. Set up the environment
```cmd
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure Credentials
Create a .env file in the root directory and add your Google API Key: 
```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Run the Application
```cmd
streamlit run chat_app/chat_frontend.py
```

## Project Structure
* chat_app/: Main Streamlit frontend and session management.
* utils/: Core logic including LangGraph definitions (using_langgraph.py) and LangChain sequences.
* rag/: Components for document indexing and retrieval using scikit-learn embeddings.
* chat/: Local storage for session history (JSON).
