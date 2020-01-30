from deep_learning.lib.utility import *
from deep_learning.preprocess.augment import *
from deep_learning.preprocess.dataframe import *

class PreProcess:
    def __init__(self,params):
        super(PreProcess,self).__init__()
        self.logger = logging.getLogger(params['logger_name']+'.preprocess')
        self.params = params
        self.aug = None
        self.df = None
    
    def process_dataframe(self):
        self.aug = Augment(self.params)
        self.aug.process_augment()
        self.df = Dataframe(self.params)
        self.df.process_dataframe()