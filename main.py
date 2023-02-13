import aiofiles
from tempfile import NamedTemporaryFile
from similarity import train_tfidf_from_csv, similarity_score
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from config import settings
from api.v1.models import Address

app = FastAPI()

UPLOAD_FOLDER = settings.upload_folder
ALLOWED_EXTENSIONS = settings.allowed_extensions


def is_file_allowed(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post('/check_duplication/')
def check_duplication(item: Address):
    """
    Check if the address is duplicated or not.
    Returns similarity value and a boolean to indicate
    whether the score is above threshold or not.
    *True* means address is duplicated.
    """
    address = item.dict(by_alias=True)
    return similarity_score([address])


@app.post('/training/')
async def train_tfidf(file: UploadFile = File(...)):
    """
    Receives a *.csv file. trains the TF-IDF models and pkl them
    """
    try:
        contents = await file.read()
        async with aiofiles.open(f"{UPLOAD_FOLDER}/{file.filename}", 'wb') as f:
            await f.write(contents)
    finally:
        await file.close()

    train_tfidf_from_csv(data_path=f"{UPLOAD_FOLDER}/{file.filename}")
    return {'message': f"File '{file.filename}' processed"}


@ app.post('/training/')
def check_training_status():
    pass


@ app.get('health-check')
def health_check():
    return {'message': 'Healthy'}


if __name__ == '__main__':
    app.run()
