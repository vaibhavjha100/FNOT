'''
Model Builder module bulids the ML models

The module contains the ModelBuilderService class which activates the model building pipeline
'''

import warnings
warnings.filterwarnings("ignore")

from loguru import logger

from config.config import settings


from model.pipeline.model import build_model


class ModelBuilderService:
    '''
    A sevice class of building ML models

    Methods:
    init - constructor to initialize objects and contains model, scaler and ticker info
    train_model - trains a ML model
    '''
    
    def __init__(self):
        '''
        Function to initialize the class
        '''
        pass
    
    def train_model(self): 
        '''
        Function to train a model
        '''
        logger.info(f'Training a model for {settings.ticker} to be saved at {settings.model_path}/{settings.model_name}.keras')
        build_model()
        
    
#test
#ml_svc = ModelService()
#ml_svc.load_model_and_scaler()
#pred=ml_svc.predict()
#print(pred)