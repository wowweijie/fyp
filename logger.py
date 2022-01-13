import logging 
import concurrent.futures 
import os

class Logger:
    def __init__(self, fileName=None):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.fileName = fileName 
        if fileName is not None:
            if not os.path.exists('./logs/'):
                os.makedirs("logs")     
            log_path = './logs/' + fileName + '.log'
            self.logger = logging.getLogger(fileName)
            self.logger.addHandler(logging.FileHandler(log_path, encoding='utf8'))
        else:
            self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

    def info(self, msg, *args):
        self.executor.submit(self.logger.info, msg, *args)