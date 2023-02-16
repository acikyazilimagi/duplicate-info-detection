from typing import Any
from pydantic import BaseModel, Field


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
