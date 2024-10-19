'''
Configuration module is used to maintain configurations like variable inputs and logs for the project.

The module contains a Settings class for maintaining variable inputs and
a logger for maintaining application logs.
'''

import warnings
warnings.filterwarnings("ignore")

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath
from loguru import logger


class Settings(BaseSettings):
    '''
    Class is used to include the mentioned variable inputs where the inputs are provided in .env file
    '''
    model_config=SettingsConfigDict(env_file='config/.env', env_file_encoding='utf-8')

    #ticker are as per yahoo finance
    ticker: str
    model_name: str
    #DirectoryPath is used to validate for directory
    model_path: DirectoryPath
    #period calues can be 1mo or 1w or 1d
    period: str


settings = Settings()

#logger maintains logs for 6 months in 1 week interval zip files
logger.add('logs/app.log', rotation='1 week', retention='6 months', compression='zip')