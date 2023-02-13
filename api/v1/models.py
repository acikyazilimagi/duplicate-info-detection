from typing import Any
from pydantic import BaseModel, Field


class Address(BaseModel):
    İl: str
    İlçe: str
    Mahalle: str
    Adres: str
    Bulvar_Cadde_Sokak_Yol_Yanyol: str = Field(
        ..., alias='Bulvar/Cadde/Sokak/Yol/Yanyol'
    )
    Bina_Adı: str = Field(..., alias='Bina Adı')
    Dış_Kapı__Blok_Apartman_No: Any = Field(...,
                                            alias='Dış Kapı/ Blok/Apartman No')
    Kat: Any
    İç_Kapı: Any = Field(..., alias='İç Kapı')
    Ad_Soyad: str = Field(..., alias='Ad-Soyad')
    Kaynak: Any
    Telefon: Any
    Oluşturulma_Tarihi: Any = Field(..., alias='Oluşturulma Tarihi')
    Güncellenme_Tarihi: Any = Field(..., alias='Güncellenme Tarihi')
    id: str


class Score(BaseModel):
    id: str
    text_score: str
    name_score: str
    is_simiar: bool
