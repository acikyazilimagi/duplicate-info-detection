import json
import os
from similarity import train_tfidf_from_csv, similarity_score
from fastapi import FastAPI
from api.v1.models import Address

app = FastAPI()

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post('/check_duplication/')
def check_duplication(item: Address):
    address = item.dict(by_alias=True)
    return json.dumps(similarity_score([address]))


@app.route('/train_tfidf/', methods=['POST'])
def train():
    if request.method == 'POST':
        """POST request receives a *.csv file. trains the TF-IDF models and pkl them"""
        # check if the post request has the file part
        if 'file' not in request.files:
            return Response("No file uploaded", status=400)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
        train_tfidf_from_csv(data_path=path)
        return Response("Models trained and saved to server")
    # train_tfidf_from_csv()
    # return Response("Models trained and saved to server")


if __name__ == '__main__':
    app.run()
