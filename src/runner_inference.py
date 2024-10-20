'''
Runner module is used for running the prediction part of the application

The module has a main function for creating a model service object and executing the methods for price prediction
'''
import warnings
warnings.filterwarnings("ignore")

from loguru import logger

from model.model_inference import ModelInferenceService


#for exceptions use catch
@logger.catch
def main():
    '''
    Function to create a ModelService object, 
    load/build the model and
    forecast stock price
    '''
    ml_svc = ModelInferenceService()
    ml_svc.load_model_and_scaler()
    pred=ml_svc.predict()
    logger.info(f'Expiry price: {pred}')

if __name__=='__main__':
    main()