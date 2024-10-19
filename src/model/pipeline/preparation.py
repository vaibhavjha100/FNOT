'''
The preparation module preprocesses stock price data for model training

The module contains prepare_data function that retuns preprocessed data and scale_data function to scale stock data
'''
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from loguru import logger

from model.pipeline.collection import load_data


def prepare_data():
    '''
    Function for data preprocessing
    '''
    logger.info('Starting the preprocessing pipeline')
    #load stock data using function for collection module
    data = load_data()
    #scale the data with scale_data function. scaler is also returned so that it can be saved with the model
    scaled_data, scaler = scale_data(data)

    return scaled_data, scaler

def scale_data(data):
    '''
    Function to scale the data using MinMaxScaler
    '''
    logger.info('Scaling adjusted close data using MinMaxScaler')
    #MinMaxScaler is used for good results with LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    #Adjusted Close is fitted to account for stock splits, dividends, etc
    scaler.fit(data['Adj Close'].values.reshape(-1,1))
    scaled_data = scaler.transform(data['Adj Close'].values.reshape(-1,1))
    return scaled_data, scaler

#test
#df=prepare_data()
#print(df)