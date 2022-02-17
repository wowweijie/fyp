import logging 
import concurrent.futures 
import os
import sys

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    
    def __init__(self, sessionName=None, timestamp=None, fileName=None):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        dirName = sessionName + '_' + timestamp if timestamp is not None else sessionName
        if fileName is None:
            fileName = "program"
        if sessionName is not None:
            if not os.path.exists('./logs/' + dirName):
                os.makedirs("logs/" + dirName)     
            self.log_path = './logs/' + dirName + '/' + fileName + '.log'
            self.logger = logging.getLogger(sessionName)
            fileHandler = logging.FileHandler(self.log_path, encoding='utf8')
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
        self.executor.submit(self.logger.info, msg, *args)