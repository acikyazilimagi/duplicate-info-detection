from similarity import process_csv
from db.crud import update_task_status, create_task
from db.database import get_db
from fastapi import Depends
from .api.v1.schemas import TaskStatus
from sqlalchemy.orm import Session


def deduplicate_csv(task_id, db: Session = Depends(get_db)):
    filename = f"{task_id}.csv"
    create_task(
        db, task_id, filename, TaskStatus.PROCESSING
    )
    with open(filename, mode='w') as output_file:
        output_file.write(process_csv(filename))
    update_task_status(task_id, TaskStatus.COMPLETED)
