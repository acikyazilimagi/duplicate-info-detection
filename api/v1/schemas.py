from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


class Address(BaseModel):
    id: str
    il: str = Field(..., alias='İl')
    ilce: str = Field(..., alias='İlçe')
    mahalle: str = Field(..., alias='Mahalle')
    adres: str = Field(..., alias='Adres')
    bulvar_cadde_sokak_yol_yanyol: str = Field(
        ..., alias='Bulvar/Cadde/Sokak/Yol/Yanyol'
    )
    bina_adi: str = Field(..., alias='Bina Adı')
    dis_kap_no: Any = Field(...,
                            alias='Dış Kapı/ Blok/Apartman No')
    kat: Any = Field(..., alias='Kat')
    ic_kapi: Any = Field(..., alias='İç Kapı')
    ad_soyad: str = Field(..., alias='Ad-Soyad')
    kaynak: Any = Field(..., alias='Kaynak')
    telefon: Any = Field(..., alias='Telefon')
    olusturulma_tarihi: Any = Field(..., alias='Oluşturulma Tarihi')
    guncellenme_tarihi: Any = Field(..., alias='Güncellenme Tarihi')


class Score(BaseModel):
    id: str
    text_score: str
    name_score: str
    is_simiar: bool


class TaskStatus (str, Enum):
    STARTED = 'started'
    PROCESSING = 'processing'
    COMPLETED = 'completed'


class CsvResponse (BaseModel):
    task_id: str
    status: TaskStatus
    file_name: str


# ORM
class CsvTask (BaseModel):
    id: str
    filename: str
    status: TaskStatus
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True
