import json
import os

from flask import Flask, request, Response, jsonify
from similarity import train_tfidf_from_csv, similarity_score


app = Flask(__name__)

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/get_scores/', methods = ['POST'])
def scoring():
    if request.method == 'POST':
        """GET request handles the case of scoring entries"""
        result = json.dumps(similarity_score(request.json))
        return Response(result ,status=200)

@app.route('/train_tfidf/', methods = ['GET'])
def train():
    if request.method == 'POST':
        """POST request receives a *.csv file. trains the TF-IDF models and pkl them"""
        # check if the post request has the file part
        if 'file' not in request.files:
            print(1111)
            return Response("No file uploaded", status=400)
        else:
            print('ELSE')
        print('ggegege')
        file = request.files['file']
        if file and allowed_file(file.filename):
                filename = file.filename
                print(000)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
        train_tfidf_from_csv(data_path=path)
        return Response("Models trained and saved to server")
    # train_tfidf_from_csv()
    # return Response("Models trained and saved to server")


if __name__ == '__main__':
    app.run()
