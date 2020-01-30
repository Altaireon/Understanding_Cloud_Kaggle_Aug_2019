from deep_learning.lib.utility import *
from deep_learning.postprocess.classify import *
from deep_learning.postprocess.segment import *

class PostProcess:
    def __init__(self,params,encoding,decoding,names):
        
        super(PostProcess,self).__init__()

        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.preprocess')
        self.params = params
        self.classify = Classify(self.params,names=names)
        self.encoding = encoding
        self.decoding = decoding
        self.segment = Segment(self.params,encoding=encoding,names=names)
    
    def process_all(self,x,ids):
        labels,_,_,_ = self.classify.process(x,ids)
        self.segment.classify = labels
        return self.segment.process(x,ids)
    
    def process_classify(self,x,ids):
        return self.classify.process(x,ids)
    
    def process_segment(self,x,ids):
        return self.segment.process(x,ids)
    
    def run_check_postprocess(self):
        self.segment.__run_check_process__()
        self.classify.__run_check_process__()