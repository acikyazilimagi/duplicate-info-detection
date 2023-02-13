from pydantic import BaseSettings


class Config(BaseSettings):
    train_api_key: str = "very-secret-api-key"
    query_api_key: str = "very-secret-other-api-key"
    duplicate_threshold: int = 80
    upload_folder: str = 'data'
    allowed_extensions: list = ['csv']


settings = Config()
