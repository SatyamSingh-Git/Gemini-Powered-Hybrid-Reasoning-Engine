import os
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
# This allows us to use os.getenv() or Pydantic's BaseSettings to read them.
load_dotenv()


class Settings(BaseSettings):
    """
    Pydantic model to manage application settings and secrets.
    It automatically reads from environment variables.
    """
    # The Field(..., env=...) is redundant if the variable name matches,
    # but it makes the source of the setting explicit.
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")

    class Config:
        # Specifies the .env file to be used
        env_file = ".env"
        env_file_encoding = "utf-8"
        # The extra='ignore' parameter tells Pydantic to ignore any extra
        # environment variables that are not defined in this model.
        extra = "ignore"


def get_settings() -> Settings:
    """
    Dependency function to get the application settings.
    This can be used in FastAPI for dependency injection.
    """
    return Settings()