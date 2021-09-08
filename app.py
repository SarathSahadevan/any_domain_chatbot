from flask import Flask, flash, request, redirect, url_for, render_template
import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
import os

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './data/pdf/'
ALLOWED_EXTENSIONS = {'txt', 'pdf','csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



#df = pd.read_csv('./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
#df = filter_paragraphs(df)
#df.head()

#cdqa_pipeline = QAPipeline(reader='./models/bert_qa_vCPU-sklearn.joblib')
#cdqa_pipeline.fit_retriever(df=df)
# global df
# global cdqa_pipeline
@app.route("/")
def home():
    return render_template("home.html")




def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#
#def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# https://stackoverflow.com/questions/18334717/how-to-upload-a-file-using-an-ajax-call-in-flask
# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
@app.route('/upload_file', methods=['POST'])
def upload_file():
    print("inside upload file")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            f = request.files['file']  
            print("Saving uploaded file")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

            print("Setting up new QA model ... ")
            global df 
            # df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), converters={'paragraphs': literal_eval})
            df = pdf_converter(directory_path=app.config['UPLOAD_FOLDER'])
            df = filter_paragraphs(df)
            global cdqa_pipeline
            print("Fitting new model ...")
            cdqa_pipeline = QAPipeline(reader='./models/bert_qa_vCPU-sklearn.joblib')
            cdqa_pipeline.fit_retriever(df=df)
            print("Redirecting back to the webapp")
            return "success"
        return "ok"


@app.route("/get")
def get_bot_response():
    global cdqa_pipeline
    userText = request.args.get('msg')
    return str(cdqa_pipeline.predict(userText)[0])

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000, debug=True)
