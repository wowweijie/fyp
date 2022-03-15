import logging 
import concurrent.futures 
import os
import sys

from torch import save
from trainer.maml.metalearners.maml_trpo import MAMLTRPO

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    
    def __init__(self, sessionName=None, timestamp=None, fileName=None):
        dirName = sessionName + '_' + timestamp if timestamp is not None else sessionName
        if fileName is None:
            fileName = "program"
        if sessionName is not None:
            if not os.path.exists('./logs/' + dirName):
                os.makedirs("logs/" + dirName)     
            self.log_dir = './logs/' + dirName
            self.logger = logging.getLogger(sessionName)
            fileHandler = logging.FileHandler(self.log_dir + '/' + fileName + '.log', encoding='utf8')
            formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
            fileHandler.setFormatter(formatter)
            self.logger.addHandler(fileHandler)
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(formatter)
            self.logger.addHandler(consoleHandler)
        else:
            self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def info(self, msg, *args):
        self.logger.info(msg, *args)
    
    def error(self, msg, *args):
        self.logger.error(msg, *args)

    def csv(self, df, filename):
        """logs a dataframe to csv format in log directory

        Args:
            df (_type_): dataframe
            filename (_type_): filename with .csv included 
        """        
        df.to_csv(self.log_dir + '/' + filename)
    
    def save_model(self, model):
        if isinstance(model, MAMLTRPO):
            with open(self.log_dir + '/maml_policy', 'wb') as f:
                save(model.policy.state_dict(), f)
        else:
            model.save(self.log_dir + '/' + type(model).__name__)
