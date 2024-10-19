'''
The model modules trains models for stock price prediction

The module contains buid_model function which is the pipeline for model building,
get_xy function which divides preprocessed data into x and y variables
split function to split train-test-validation data,
design_model function that creates the architecture for a LSTM model,
get_hyperparameters function that returns the best hyperparameters for the LSTM model,
train_model function that trains the model,
evaluate_model function that tests the data with mean absolute error,
save_model that saves the model
and acceptable_deviation function that returns an acceptable mean absolute error for the model 
'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense,LSTM,Dropout # type: ignore
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import keras_tuner as kt
import joblib
from loguru import logger

from model.pipeline.preparation import prepare_data
from config.config import settings


def build_model(create_new_parameters = False):
    '''
    Function is a machine learning pipeline that divides data into x & y, splits that into train-test,
    creates best hyperparameters, builds model architecture, trains the model and evaluates the model.
    If the score of the model is acceptable, the model and the scaler are saved,
    else the function is initiated again with new parameters.
    '''
    logger.info('Starting model building pipeline')
    data, scaler = prepare_data()
    x, y = get_xy(data)
    xtr, xtt, ytr, ytt = split(x, y)
    #if new hyperparameters are to be created
    if create_new_parameters==True:
        hp = get_hyperparameters(xtr, ytr, create_new = True)
    #if existing hyperparameters are used
    else:
        hp = get_hyperparameters(xtr, ytr)
    designed_model = design_model(hp)
    model = train_model(xtr, ytr, designed_model)
    score = evaluate_model(model, xtt, ytt, scaler)
    if score < acceptable_deviation():
        save_model(model)
        logger.info(f'Saving scaler to {settings.model_path}/{settings.model_name}_scaler.joblib')
        joblib.dump(scaler, f'{settings.model_path}/{settings.model_name}_scaler.joblib')
    else:
        logger.warning('Unsatisfactory model. Initiate pipeline with new hyperparameters')
        build_model(create_new_parameters = True)

def get_xy(data, timestep = 60):
    '''
    Function returns independent variable x and dependent variable y
    x is the timestep days adjusted close prices before y
    '''
    logger.info(f'defining x and y variables where x is {timestep} days adjusted close before y')
    x, y = [], []
    for i in range(len(data)-timestep-1):
        a = data[i:(i+timestep), 0]
        x.append(a)
        y.append(data[i + timestep, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

def split(x,y,test_size=0.2):
    '''
    Function to perform train-test split
    There is no shuffle to preverse the time series nature of the data
    '''
    logger.info('splitting train & test/validation data')
    return train_test_split(x, y, test_size = test_size, random_state = 42, shuffle = False)
    
def design_model(hp):
    '''
    Functions returns a model architecture for an LSTM model with 2 LSTM layersand 1 dense layer
    Hyperparameters are fed to the function for making the architecture
    Dropout is added to prevent overfitting
    '''
    logger.info('Designing model architecture')
    model = Sequential()
    #60 in the input shape is timestep from get_xy function
    model.add(LSTM(units = hp.Int('units_1', min_value = 30, max_value = 300, step = 30), return_sequences = True, input_shape = (60,1)))
    if hp.Boolean("dropout1"):
        model.add(Dropout(0.1))
    model.add(LSTM(units = hp.Int('units_2', min_value = 20, max_value = 200, step = 20), return_sequences = False))
    if hp.Boolean("dropout2"):
        model.add(Dropout(0.1))
    #add a dense layer with 25 units. after lstm has done feature extraction, this layer helps to refine and adjust features
    #for final prediction
    model.add(Dense(units = hp.Int('units_d', min_value = 5, max_value = 50, step = 5)))
    #output
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model

def get_hyperparameters(x, y, create_new=False):
    '''
    Function retuns the best hyperparameters
    x & y are taken as arguments
    create_new argument is for overwriting previous hyperparameters
    RandomSearch is used for the best possible result by sacrificing time to tune
    '''
    logger.info('Generating best hyperparameters')
    #split is done for 70% train and 30% validation data
    xtr, xtt, ytr, ytt= split(x, y, test_size = 0.3)
    if create_new == True:
        tuner=kt.RandomSearch(
            hypermodel = design_model,
            objective = 'val_loss',
            max_trials = 5,
            executions_per_trial = 2,
            overwrite = True,
            directory = 'model/tuning_dir',
            project_name = 'lstm_tuning'
        )
    else:
        tuner = kt.RandomSearch(
            hypermodel = design_model,
            objective = 'val_loss',
            max_trials = 5,
            executions_per_trial = 2,
            directory = 'model/tuning_dir',
            project_name = 'lstm_tuning'
        )
    tuner.search(xtr, ytr, epochs = 50, validation_data = (xtt, ytt))
    best_hps = tuner.get_best_hyperparameters(5)

    return best_hps[0]

def train_model(x, y, model):
    '''
    Function to train the lstm model
    '''
    logger.info('Training model')
    #split is done for 80% train and 20% validation data
    xtr, xtt, ytr, ytt = split(x, y)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, start_from_epoch = 50)
    model.fit(xtr, ytr, batch_size = 64, epochs = 100, validation_data = (xtt, ytt), callbacks = [early_stopping])

    return model

'''
#old trainer
def train_model(x,y):
    xtr, xtt, ytr, ytt= split(x,y)
    #early stopping for when validation loss is not improving. we start this from 50th epoch and wait for 5 epochs to observe if val
    #loss decreases
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, start_from_epoch=50)
    #building an LSTM model
    model=Sequential()
    #adding the 1st layer with 150 units. returns output for next layer and takes input in training shape
    model.add(LSTM(units=150, return_sequences=True, input_shape=(xtr.shape[1],1)))
    #add dropout to prevent overfitting by randomly setting 10% of the input as 0
    model.add(Dropout(0.1))
    #2nd LSTM layer with 90 units. don't return anything as it is the last lstm layer
    model.add(LSTM(units=90, return_sequences=False))
    #add another 10% dropout
    model.add(Dropout(0.1))
    #add a dense layer with 25 units. after lstm has done feature extraction, this layer helps to refine and adjust features
    #for final prediction
    model.add(Dense(units=25))
    #output
    model.add(Dense(units=1))
    #model compiled with adam optimizer and loss measure as mse
    model.compile(optimizer='adam',loss='mean_squared_error')
    #fit model to training data for 100 epochs with 64 batch size. validation data is testing data with early stopping
    model.fit(xtr, ytr, batch_size = 64, epochs = 100, validation_data=(xtt,ytt), callbacks=[early_stopping])

    return model
'''

def evaluate_model(model, xtt, ytt, scaler):
    '''
    Function to test the model with MAE
    '''
    logger.info('Testing model')
    yp = model.predict(xtt)
    #rescale to og prices
    yp = scaler.inverse_transform(yp.reshape(-1,1))
    ytt = scaler.inverse_transform(ytt.reshape(-1,1))
    #mean absolute error is used so the price deviation can be seen against the financial instrument prices
    logger.info(f'Evaluation result (MAE): {mean_absolute_error(ytt,yp)}')
    return mean_absolute_error(ytt,yp)

def save_model(model):
    '''
    Function to save the model to model path
    '''
    logger.info(f'Saving model to {settings.model_path}/{settings.model_name}.keras')
    model.save(f'{settings.model_path}/{settings.model_name}.keras')

def acceptable_deviation():
    '''
    Function to return the acceptable MAE for the model
    0.6% deviation is provided
    '''
    #item() is used to get the float value out of the latest price of the financial instrument
    return 0.006 * yf.download(settings.ticker,period='1d')['Adj Close'].item()


#test
#build_model()