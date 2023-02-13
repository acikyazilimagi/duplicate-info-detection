from pydantic import BaseSettings


class Settings(BaseSettings):
    train_api_key: str = "very-secret-api-key"
    query_api_key: str = "very-secret-other-api-key"
    duplicate_threshold: int = 80
    upload_folder: str = 'data'
    allowed_extensions: list = ['csv']

    class Config:
        env_file = ".env"


settings = Settings()
