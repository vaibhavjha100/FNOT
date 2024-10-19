'''
The collection modules collects stock price data for specified ticker(yahoo finance)

Module contains load_data function that loads daily stock price data for the specified ticker
'''
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from loguru import logger

from config.config import settings


def load_data():
    '''
    Function to return daily stock price data for maximum interval
    '''
    logger.info(f'Downloading daily stock price data for {settings.ticker}')
    return yf.download(settings.ticker,period = 'max', interval = '1d')

#test
#df=load_data('^NSEI')
#print(df)