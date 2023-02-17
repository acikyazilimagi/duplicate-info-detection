import aiofiles
from similarity import train_tfidf_from_csv, similarity_score
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks, Response, Request, HTTPException
from fastapi.responses import FileResponse
from config import settings
from api.v1.schemas import Address, CsvResponse
from auth import authorize_trainer, authorize_checker
from tasks import deduplicate_csv as deduplicate_task
from db.database import SessionLocal
from db.crud import get_task, delete_task

app = FastAPI()

UPLOAD_FOLDER = settings.upload_folder
ALLOWED_EXTENSIONS = settings.allowed_extensions


def is_file_allowed(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    response = Response("Internal server error", status_code=500)
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        request.state.db.close()
    return response


@app.post('/check_duplication/')
def check_duplication(item: Address, token: str = Depends(authorize_checker)):
    """
    Check if the address is duplicated or not.
    Returns similarity value and a boolean to indicate
    whether the score is above threshold or not.
    *True* means address is duplicated.
    """
    address = item.dict(by_alias=True)
    return similarity_score([address])


@app.post('/training/')
async def train_tfidf(file: UploadFile = File(...), token: str = Depends(authorize_trainer)):
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


@app.post('/csv/')
async def deduplicate_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...), token: str = Depends(authorize_checker)):
    """
    Receives a *.csv file. deduplicates the data.
    """
    try:
        contents = await file.read()
        async with aiofiles.open(f"{UPLOAD_FOLDER}/{file.filename}", 'wb') as f:
            await f.write(contents)
    finally:
        await file.close()

    background_tasks.add_task(deduplicate_task)

    return CsvResponse(
        task_id=f"{file.filename}",
        status="started",
        file_name=f"{file.filename}"
    )


@app.get('/csv/{task_id}/download')
async def get_csv(task_id: str, token: str = Depends(authorize_checker)):
    """
    Receives the deduplicated *.csv file created by task with the task_id
    """
    return FileResponse(f"./{task_id}.csv", filename=f"{task_id}.csv")


@app.get('/csv-task/{task_id}')
def check_task_status(task_id):
    task = get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return CsvResponse(
        task_id=f"{task.id}",
        status=task.status,
        file_name=f"{task.filename}"
    )


@app.delete('/csv/{task_id}')
def delete_task(task_id):
    task = get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    delete_task(task_id)

    return {'message': f"Task '{task_id}' deleted"}


@app.get('health-check')
def health_check():
    return {'message': 'Healthy'}


if __name__ == '__main__':
    app.run()
