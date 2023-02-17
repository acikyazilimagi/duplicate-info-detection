from sqlalchemy.orm import Session
from . import models


def get_task(db: Session, task_id: int):
    return db.query(models.CsvTask).filter(models.CsvTask.id == task_id).first()


def update_task_status(db: Session, task_id: int, status):
    db.query(models.CsvTask).filter(
        models.CsvTask.id == task_id).update({'status': status})
    db.commit()


def delete_task(db: Session, task_id: str):
    db.query(models.CsvTask).filter(models.CsvTask.id == task_id).delete()
    db.commit()


def create_task(db: Session, task_id: str, filename: str, status: str):
    db_task = models.CsvTask(id=task_id, filename=filename, status=status)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task
