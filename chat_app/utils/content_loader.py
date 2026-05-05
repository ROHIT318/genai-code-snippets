from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from typing import List
import os

class ContentLoader():
    def __init__(self, url_list: List[str]):
        self.url_list = url_list
        self.url_doc_contents = self.url_document_loader()
        self.clean_document_content = self.document_cleaner()

    def url_document_loader(self):
        url_or_pdf = ''

        for url in self.url_list:
            if(url.startswith('https') or url.startswith('http')):
                url_or_pdf = 'url'
                break

        if url_or_pdf=='url':
            wb_loader = WebBaseLoader(web_path=self.url_list)
            return wb_loader.load()
        else:
            pdf_doc_content = []
            for url in self.url_list:
                pdf_loader = PyPDFLoader(url)
                pdf_doc_content.extend(pdf_loader.load())
            return pdf_doc_content
    
    def document_cleaner(self):
        clean_doc = []
        for doc in self.url_doc_contents:
            # doc.metadata['title'] = doc.metadata['title'].replace('\n', '').replace(' ', '')
            doc.page_content = doc.page_content.replace('\n', '').replace('  ', '')
            clean_doc.extend(doc)
        return clean_doc

    def display_all_documents(self):
        for doc in self.clean_document_content:
            print(doc)

    
if __name__ == '__main__':
    
    ############################################
    #           JOB Page Loading 1             #
    ############################################
    # url_list = ['https://usijobs.deloitte.com/en_US/careersUSI/JobDetail/USI-EH26-EA-GTLO-Sharepoint-Senior-Analyst/340662', 
    #             'https://usijobs.deloitte.com/en_US/careersUSI/JobDetail/USI-EH26-EA-GTLO-Analytics-Insight-Senior-Analyst/340661', 
    #             'https://usijobs.deloitte.com/en_US/careersUSI/JobDetail/USI-EH26-TIS-Digital-Analytics-Skills-Lead-Skills-Transformation-Solutions-Workstream-Assistant-Manager/340519'
    #             ]
    # content_loader = ContentLoader(url_list)
    # content_loader.display_all_documents()

    ############################################
    #           JOB Page Loading 2             #
    ############################################
    url_list = ['https://www.pwc.in/careers/experienced-jobs/description.html?wdjobreqid=280138WD&wdcountry=IND&jobtitle=Associate&wdjobsite=Global_Experienced_Careers&wdjd=simple', 
                'https://www.pwc.in/careers/experienced-jobs/description.html?wdjobreqid=288872WD&wdcountry=IND&jobtitle=1-10yrs+Application+for+Cyber-+Kolkata+DN+57+-+RDC&wdjobsite=Global_Experienced_Careers&wdjd=simple', 
                'https://www.pwc.in/careers/experienced-jobs/description.html?wdjobreqid=439388WD&wdcountry=IND&jobtitle=Associate&wdjobsite=Global_Experienced_Careers&wdjd=simple'
                ]
    content_loader = ContentLoader(url_list)
    content_loader.display_all_documents()

    # ############################################
    # #              PDF Loading                 #
    # ############################################
    # uploaded_pdf_loader = ContentLoader(url_list=[os.path.abspath('files/gradient_descent.pdf')])
    # uploaded_pdf_loader.display_all_documents()