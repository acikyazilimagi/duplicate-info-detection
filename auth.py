
from enum import Enum
from fastapi.security.api_key import APIKeyHeader
from fastapi import Depends, HTTPException
from starlette import status
from config import settings


class KeyType(Enum):
    TRAINING = 'training'
    QUERY = 'query'


api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def authorize_trainer(api_key_header: str = Depends(api_key_header)):

    if api_key_header != f"Token {settings.train_api_key}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate API KEY"
        )


async def authorize_checker(api_key_header: str = Depends(api_key_header)):
    print(api_key_header)
    if api_key_header != f"Token {settings.query_api_key}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate API KEY"
        )
