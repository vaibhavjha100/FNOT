'''
Runner buiilder module is used for running the model building part of the application

The module has a main function for creating a model builder service object and executing the methods for model training
'''
import warnings
warnings.filterwarnings("ignore")

from loguru import logger

from model.model_builder import ModelBuilderService


#for exceptions use catch
@logger.catch
def main():
    '''
    Function to create a ModelService object, 
    load/build the model and
    forecast stock price
    '''
    logger.info("Starting the model building process")
    ml_b = ModelBuilderService()
    ml_b.train_model()

if __name__=='__main__':
    main()