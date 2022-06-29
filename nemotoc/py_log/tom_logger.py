import logging
import os
class Log:
    '''
    returned logging class
    '''
    def __init__(self, logger = None):
        '''
        define the path/the level/creation of log files
        '''
        self.logFile = '%s/%s'%(os.getcwd(), 'polysome.log')  
        #create the logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.INFO)
        #create filehander
        fh = logging.FileHandler(self.logFile, 'a')
        fh.setLevel(logging.INFO)
        #create consolehander 
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)        
        #define the format 
        formatter = logging.Formatter(
                '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        #add handler to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        #close handler
        fh.close()
        ch.close()
        
    def getlog(self):
        return self.logger     