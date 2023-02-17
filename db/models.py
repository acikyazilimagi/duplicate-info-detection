from .database import Base
from sqlalchemy import TIMESTAMP, Column, String
from sqlalchemy.sql import func
from fastapi_utils.guid_type import GUID, GUID_DEFAULT_SQLITE


class CsvTask(Base):
    __tablename__ = 'csv_tasks'
    id = Column(GUID, primary_key=True, default=GUID_DEFAULT_SQLITE)
    filename = Column(String, nullable=False)
    status = Column(String, nullable=False)
    createdAt = Column(TIMESTAMP(timezone=True),
                       nullable=False, server_default=func.now())
    updatedAt = Column(TIMESTAMP(timezone=True),
                       default=None, onupdate=func.now())
