'''
Model Inference Service module uses the built ML models

The module contains the ModelInferenceService class which contains the features to load and use ml models,
last_thursday function to get the next last thurday of the month and
next_thurday function to get coming thursday
'''
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from loguru import logger

from config.config import settings


from model.pipeline.model import build_model


class ModelInferenceService:
    '''
    A sevice class of managing ML models

    Methods:
    init - constructor to initialize objects and contains model, scaler and ticker info
    load_model_and_scaler - it loads model and scaler
    predict - forcasts the price either on a weekly or monthly basis
    '''
    
    def __init__(self):
        '''
        Function to initialize the class
        '''
        self.model=None
        self.scaler=None
        self.ticker=settings.ticker
    
    def load_model_and_scaler(self): 
        '''
        Function to load model and scaler
        '''
        logger.info(f'Checking the existence of model config file at {settings.model_path}/{settings.model_name}.keras')
        model_path = Path(f'{settings.model_path}/{settings.model_name}.keras')
        scaler_path = Path(f'{settings.model_path}/{settings.model_name}_scaler.joblib')
        if not model_path.exists() and not scaler_path.exists():
            raise FileNotFoundError('Model file does not exist')
        logger.info('Model exists. Loading model and scaler')
        self.model=load_model(model_path)
        self.scaler=joblib.load(scaler_path)
    
    def predict(self):
        '''
        Function to forcast stock price
        '''
        logger.info('Forecasting')
        period=settings.period
        today = pd.Timestamp(datetime.now())
        year = today.year
        month = today.month
        if period=="1mo":
            expiry_date = last_thursday(year, month)
        if period=="1w" or period=="1d":
            expiry_date = next_thursday()
        #check if last thursday of this month has passed
        if today > expiry_date:
            #increment month
            expiry_date = last_thursday(year, month + 1 if month < 12 else 1)
            #increment year if december
            year = year if month < 12 else year + 1
        #get trading days for nse
        nse_calendar = mcal.get_calendar('NSE')
        #get trading schedule from today till last thurday/expiry day
        schedule = nse_calendar.schedule(start_date=today, end_date=expiry_date)
        #no, of days between today and expiry
        trading_days = mcal.date_range(schedule, frequency='1D').shape[0]
        #set end point as today
        end_date = pd.Timestamp.today()
        #start date is 200 days ago
        start_date = end_date - pd.DateOffset(days=200)
        #get trading schedule from 200 days ago till today
        schedule = nse_calendar.schedule(start_date=start_date, end_date=end_date)
        # select last 61 days in the schedule
        training_days = mcal.date_range(schedule, frequency='1D')[-61:]
        #download daily price data for the last 60-61 days
        stock_data = yf.download(self.ticker, start=training_days[0], end=training_days[-1])
        #count to adjust training days
        c=0
        #loop to ensure that exactly 60 days are retrived as our input shape is (60,1)
        while stock_data.shape[0] !=60:
            # end date is today and start is 200 days ago
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(days=200)
            #get nse schedule
            schedule = nse_calendar.schedule(start_date=start_date, end_date=end_date)
            #adjust training days for counter c
            training_days = mcal.date_range(schedule, frequency='1D')[-60-c:]
            #redownload accordint to new training days
            stock_data = yf.download(self.ticker, start=training_days[0], end=training_days[-1])
            #check if exactly 60 days are retrieved and asjust c if not
            if stock_data.shape[0] > 60:
                c-=1
            if stock_data.shape[0] < 60:
                c+=1
        x = self.scaler.transform(stock_data['Close'].values.reshape(-1,1))
        #transpose data to match input shape
        x = x.T

        ret=[]
        #loop through each trading day till expiry to make predictions
        for i in range(trading_days):
            y=self.model.predict(x)
            ret.append(y)
            #update input data x to make next prediction
            x = np.append(x,y)
            #delete 1st element in x to maintain input shape
            x = x[1:]
            x = x.reshape(1,-1)
        #rescale to og prices
        ret=self.scaler.inverse_transform(np.array(ret).reshape(-1,1))
        
        #return the next day close price for daily trading
        if period=="1d":
            return ret[0]

        #retun the price for the last forcast

        return ret[-1]


def last_thursday(year, month):
    '''Function to return the last thursday of the current month'''
    #get the last day of the month
    last_day = pd.Timestamp(datetime(year, month, 1) + pd.offsets.MonthEnd(0))
    #subtract days till u hit a thursday
    while last_day.weekday() != 3:
        last_day -= pd.Timedelta(days=1)
    return last_day

def next_thursday():
    '''Function to return the next thursday'''
    #get current date
    date = pd.Timestamp(datetime.now())
    #see how many days are left to thursday
    days_ahead = 3 - date.weekday()
    #if this thursday is passed then look for next week's thursday
    if days_ahead <= 0:
        days_ahead += 7
    #add days to thursday to current date
    next_thursday_date = date + pd.Timedelta(days=days_ahead)
    return next_thursday_date

    
#test
#ml_svc = ModelService()
#ml_svc.load_model_and_scaler()
#pred=ml_svc.predict()
#print(pred)