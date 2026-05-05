from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Annotated, Literal, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json

from content_loader import ContentLoader

load_dotenv('.env')
chat_gemini_api_key = os.getenv('chat_gemini_api_key')
str_output_parser = StrOutputParser()

# 0. node_0 is a chat prompt template which prepares the prompt to be sent to chat_model
# 1. Will classify in the chat itself whether it is an excel transformation operation or a simple query based task.
    # 1.1. To be done by node 1 and chat_model_1.
    # 1.2. It needs to produce structured output.
# 2. If it is an excel based transformation then redirect to node_2.
    # 2.1. chat_model_2 will produce a list of steps. 
    # 2.2. execute those list of steps one after another.
# 3. If it is not an excel based transformation then redirect to node_3
    # 3.1. Simple Q&A will continue.


class ChatModelSchema(BaseModel):
    """ Output schema to be produced by chat_model_1 """
    query_type: Annotated[str, Literal['excel_transformation', 'correct_job_match', 'not_excel_transformation']] = Field(description=(
        "Classify the user query into one of three categories:\n"
        "- 'excel_transformation': Query involves transforming tabular or Excel data.\n"
        "- 'correct_job_match': Query asks for job recommendations based on a resume or CV.\n"
        "- 'not_excel_transformation': Any other type of query."
    )) 
    output: List[str] = Field(description=(
        "The original query returned as a list of strings. "
        "Returns an empty list if query_type is 'excel_transformation'."
    ))


# chat_model_1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=chat_gemini_api_key)
chat_model_1 = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview', api_key=chat_gemini_api_key)
# Converting chat model so that output structure is of type ChatModelOutput
structured_chat_model_1 = chat_model_1.with_structured_output(ChatModelSchema)


# state variable for the graph
class StateSchema(TypedDict):
    prompt: str
    message_history: List[Dict[str, str]]
    # query_type: Annotated[str, Literal['excel_transformation', 'not_excel_transformation']] = Field(description='Output is produced will on the basis of prompt provided and can either be excel_transformation and not_excel_transformation.') 
    # Not doing strict type checking it is already handled by ChatModelSchema 
    query_type: str
    output: List[str]
    job_description: List[str] = Field(description='Contains job description that AI needs to analyse and score during ATS scoring.' )
    resume_details: str = Field(description='Resume details of end user.')

class JobDescription(TypedDict):
    prompt: str = Field(description='General query asked by the end user to AI Assistant. Query in this related to JOB postings.')
    job_url: List[str] = Field(description='Contains a list of url which points to different job postings.')

class ATSScoreSchema(TypedDict):
    prompt: str = Field(description='General query asked by the end user to AI Assistant.')
    # message_history: List[Dict[str, str]] = Field(description='Conversation history between AI assistant and human.')
    resume_description: str = Field(description='Contains the details of the resume shared by end user, containing education history, job experience, certifications achievements and skills.')
    job_description: Dict[str, str] = Field(description='Contains job role title as key of the dictionary and job description as value of the dictionary.')
    ats_score: Dict[str, int] = Field(description='Contains job role title as key of the dictionary and ATS score for a resume to that paricular job description as value of the dictionary.')
    job_wise_improvements: Dict[str, str] = Field(description='Contains job role title as key of the dictionary and improvements that can be done in the resume and in personal skillset with certifications to increase the score for that paricular job description to improve the ATS score, it will be the value of the dictionary.')
    overall_improvements: str = Field(description='Overall improvements that can be done in the resume, skillset and additional certifications that can be done to improve the ATS score.')


def node_0(input_state: StateSchema):
    query = input_state['prompt']
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful AI assistant. Answer the queries from user." '),
        ('human', '{query}')
    ])
    input_state['prompt'] = prompt_template.invoke({'query': query})
    print('Executed Node 0 !!')
    return input_state


def node_1(input_state: StateSchema):
    res = structured_chat_model_1.invoke(input_state['prompt'])
    input_state['query_type'] = res.query_type
    print('Executed Node 1 !!')
    return input_state


# chat_model_2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=chat_gemini_api_key)
chat_model_2 = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview', api_key=chat_gemini_api_key)
def node_2(input_state: StateSchema):
    query = input_state['prompt']
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful AI assistant that provides back the python code for transformation of a pandas dataframe based on user query. Provide only the python code and nothing else. Each python code line will be seaparated by a comma.'),
        ('human', '{query}')
    ])
    # Want to store my final prompt
    input_state['prompt'] = prompt_template.invoke(query)
    input_state['output'] = (chat_model_2 | str_output_parser).invoke(input_state['prompt'])
    print('Executed Node 2 !!')
    return input_state

# Needs further improvement
chat_model_2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=chat_gemini_api_key)
jd_chat_model = chat_model_2.with_structured_output(JobDescription)
ats_chat_model = chat_model_2.with_structured_output(ATSScoreSchema)
def node_3(input_state: StateSchema, output_state: ATSScoreSchema):

    output = jd_chat_model.invoke(input_state['prompt'])
    print(f'output: {output}')

    content_loader = ContentLoader(output['job_url'])
    input_state['job_description'] = content_loader.clean_document_content

    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'You are an helpful AI Assisstant, your job is to extract the different url provided in input prompt and return them in a form of list.'),
        ('human', 'Here is the asked by the end user: {query}. {job_description}  are the one that user is interested in. Resume of end user {resume}.')
    ])
    output_state = (prompt_template | ats_chat_model ).invoke({'query': input_state['prompt'], 'job_description': input_state['job_description'], 'resume': input_state['resume_details']})
    print(output)

    print('Executed Node 3 !!')
    return input_state, output_state


# Needs further improvement
def node_5(input_state: StateSchema, output_state: ATSScoreSchema):
    query = input_state['prompt']
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'You are an helpful AI Assisstant, act as an ATS scorer and your job is to score the resume based on different job descriptions.'),
        ('human', '{resume_description} {job_description} {message_history}')
    ])
    input_state['prompt'] = prompt_template.invoke(query)
    input_state['output'] = (ats_chat_model | str_output_parser).invoke(input_state['prompt'])
    print('Executed Node 3 !!')
    return input_state

def node_4(input_state: StateSchema):
    query = input_state['prompt']
    prompt_template = ChatPromptTemplate([
        ('system', 'You are a helpful AI chat assistant. Answer the queries of user to the best of your knowledge.'),
        ('human', '{query}')
    ])
    # Want to store my final prompt
    input_state['prompt'] = prompt_template.invoke(query)
    input_state['output'] = (chat_model_2 | str_output_parser).invoke(query)
    print('Executed Node 4 !!')
    return input_state


def conditional_node(input_state: StateSchema):
    if input_state['query_type']=='excel_transformation':
        return "node_2"
    elif input_state['query_type']=='correct_job_match':
        return "node_3"
    else:
        return "node_4"

graph = StateGraph(StateSchema)

graph.add_node('node_0', node_0)
graph.add_node('node_1', node_1)
graph.add_node('node_2', node_2)
graph.add_node('node_3', node_3)
graph.add_node('node_4', node_4)
graph.add_node('node_5', node_5)

graph.add_edge(START, 'node_0')
graph.add_edge('node_0', 'node_1')
graph.add_conditional_edges('node_1', conditional_node, {
        "node_2": "node_2",
        "node_3": "node_3",
        "node_4": "node_4",
        "node_5": "node_5",
    }
)
graph.add_edge('node_2', END)
graph.add_edge('node_3', END)
graph.add_edge('node_4', END)
graph.add_edge('node_5', END)

final_workflow = graph.compile()

if __name__ == '__main__':

    # # General Query
    # input_state = {'prompt': 'How are you?'}
    # res = final_workflow.invoke(input_state)
    # print(f'Prompt is {res}')


    # # Excel Transformation
    # input_state = {"prompt": "Add the column 'a' and 'b' in table tab to produce a final table having column 'res'. Further on, create another column called 'mul_res' which is produced by a*b. Create another column 'even_or_odd': if a is even then the new column stores 'Even' else 'Odd'."}
    # res = final_workflow.invoke(input_state)
    # print(f'Prompt is {res}')

    # ats job node testing
    input_state = {
        'prompt': 'Could you please look at this job description and let me know whether I am a goof fit for this roles? https://www.pwc.in/careers/experienced-jobs/description.html?wdjobreqid=280138WD&wdcountry=IND&jobtitle=Associate&wdjobsite=Global_Experienced_Careers&wdjd=simple https://www.pwc.in/careers/experienced-jobs/description.html?wdjobreqid=288872WD&wdcountry=IND&jobtitle=1-10yrs+Application+for+Cyber-+Kolkata+DN+57+-+RDC&wdjobsite=Global_Experienced_Careers&wdjd=simple and this one too https://www.pwc.in/careers/experienced-jobs/description.html?wdjobreqid=439388WD&wdcountry=IND&jobtitle=Associate&wdjobsite=Global_Experienced_Careers&wdjd=simple',
        'resume_details': "Rohit Sharma Data Analyst Kolkata \u2022 iro1hit@gmail.com \u2022 + 91 8777565293 \u2022 linkedin.com/in/rohit318/ \u2022 github.com/ROHIT318 SUMMARY Data Analytics professional with 3+ years of hands on experience developing and maintaining Microsoft Power BI dashboards, DAX measures, and SQL driven reporting solutions in a global, consulting oriented environment. Proven ability to apply data wrangling, ETL processes, and advanced Excel functions to surface actionable trends, patterns, and insights that drive data informed strategic decision making. Skilled in agile software development practices, dashboard testing, user acceptance processes, and cross functional collaboration under tight deadlines, with exposure to Python, machine learning, and AI concepts applied to analytics and automation workflows.PROFESSIONAL EXPERIENCE PWC INDIAKolkata, IndiaMACHINE LEARNING AND ANALYTICS DEVELOPER AUG 2022-Present \u2022 Collected, Developed and maintained 10+ Microsoft Power BI dashboards and advanced reports, embedding strong DAX expertise and best practice visual design principles to deliver data-driven, insightful solutions for leadership MOP meetings and operational teams. \u2022 Improved turnaround time (TAT) by 70% and task completion rate by 20% for the Maintenance Master Data team by building a Power BI reporting solution that identified process bottlenecks and individual performance patterns, directly supporting continuous improvement initiatives. \u2022 Reduced 300+ hours of monthly manual effort within a single financial year by developing Python (Pandas, NumPy) scripts and Power BI dashboards to automate data analysis and reporting tasks, improving reporting efficiency and standardization across deliverables. \u2022 Designed and implemented a Capacity and Demand Forecasting Tool deployed across 5 cross-functional teams, Vendor Master (On-boarding & Maintenance), D2P Materials Planning, H2R Talent Acquisition, and International Assignee using AWS SageMaker, Python (pandas, numpy, prophet), SQL, Snowflake, and Power BI executive dashboards to optimize team capacity planning. \u2022 Applied SQL, ETL processes, and data management technologies to automate multiple KPIs on the performance board after validating calculation logic with Subject Matter Experts (SMEs), ensuring data integrity, consistency, and compliance across analytics deliverables. \u2022 Conducted correlation analysis between performance metrics and underlying raw data to identify inter-metric dependencies, applying analytical and critical thinking skills to support quality assurance and reporting accuracy. \u2022 Spearheaded an end-to-end workforce study using Python, Pandas, GitLab, SharePoint, and Snowflake to forecast demand, translating complex psychometric patterns into actionable Power BI visualizations aligned with organizational strategic planning goals. \u2022 Accelerated operational decision-making by 4x through deployment of a Retrieval-Augmented Generation (RAG) system leveraging AI concepts, achieving 92% contextual alignment with official SOPs demonstrating applied knowledge of AI implementation in data analytics and automation. \u2022 Prepared and maintained internal project documentation, development standards, and reporting frameworks, ensuring consistency, quality, and compliance across all analytics projects in alignment with delivery guidelines. \u2022 Worked effectively in a global environment with changing priorities and tight deadlines, managing multiple dashboard development, testing, validation, and release readiness cycles concurrently using agile software development practices and JIRA. BUSINESS ANALYST \u25cf Created project requirement documents for 10+ analytics and reporting projects within a single year, chairing status calls to align stakeholders on requirements, UAT execution, delivery timeliness, and issue resolution, demonstrating strong verbal and written communication skills in a consulting role context. \u25cf Facilitated 8+ agile sprint retrospective sessions in JIRA, identifying root causes of project delays and implementing remedial actions that improved development task TAT by 4 days, contributing to standardized quality processes and continuous improvement. EDUCATION ST. XAVIER'S COLLEGE (AUTONOMOUS) KOLKATA, INDIAMaster of Science, Computer Science 2020-2022 UNIVERSITY OF CALCUTTA Kolkata, India Bachelor of Science, Hons in Computer Science 2017-2020 WEST BENGAL COUNCIL OF HIGHER SECONDARY EDUCATION Kolkata, India 2017 WEST BENGAL BOARD OF SECONDARY EDUCATION Kolkata, India 2015 ADDITIONAL INFORMATION \u25cf Secured top most rating for two consecutive years (FY 23 and FY 24) at PwC India. \u25cf 3 culture catalyst awards in FY26. \u25cf Certifications: PL-900, PL-300 (Certification in Microsoft Power BI), GH-300 (Microsoft GitHub Copilot), won innovation champion badge at PwC India 2022, Runner up in inter-college hackathon competition 2020 (University of Calcutta)."
    }
    # output = jd_chat_model.invoke(input_state['prompt'])
    # print(output)

    output_state = ATSScoreSchema()
    input, result = node_3(input_state, output_state)
    print(input)
    print('-----')
    print(result)