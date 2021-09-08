import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline

from cdqa.utils.download import download_model, download_bnpp_data

#download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
#download_model(model='bert-squad_1.1', dir='./models')

df = pd.read_csv('./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
df = filter_paragraphs(df)
# df.head()

cdqa_pipeline = QAPipeline(reader='./models/bert_qa_vCPU-sklearn.joblib')
cdqa_pipeline.fit_retriever(df=df)

query = 'Since when does the Excellence Program of BNP Paribas exist?'
prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))

# import os
# import pandas as pd
# from ast import literal_eval

# from cdqa.utils.converters import pdf_converter
# from cdqa.utils.filters import filter_paragraphs
# from cdqa.pipeline import QAPipeline
# from cdqa.utils.download import download_model

# # Download pdf files from BNP Paribas public news
# def download_pdf():
#     import os
#     import wget
#     directory = './data/pdf/'
#     models_url = [
#       'https://invest.bnpparibas.com/documents/1q19-pr-12648',
#       'https://invest.bnpparibas.com/documents/4q18-pr-18000',
#       'https://invest.bnpparibas.com/documents/4q17-pr'
#     ]

#     print('\nDownloading PDF files...')

#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     for url in models_url:
#         wget.download(url=url, out=directory)

# download_pdf()
# df = pdf_converter(directory_path='./data/pdf/')
# df.head()