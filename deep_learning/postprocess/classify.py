from deep_learning.lib.utility import *

class Classify:
    def __init__(self,params,names,activation=None):
        
        super(Classify,self).__init__()
        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.postprocess.classify')
        self.params = params
        if type(self.params['postprocess']['threshold_class']) == int:
            self.threshold = np.full((self.params['num_class'],),self.params['postprocess']['threshold_class'])
        else:
            self.threshold = self.params['postprocess']['threshold_class']
        self.counter = 0
        self.activation = activation
        if len(names) > 0:
            self.names=names
        else:
            self.names=range(0,self.params['num_class'])
        if self.params['inference_model']['classify_path'] != None:
            self.df_classify = pd.read_csv(LOG_DIR+self.params['inference_model']['classify_path'])
            self.df_classify.fillna('',axis=1,inplace=True)
            self.df_classify['Image'] = self.df_classify['Image_Label'].str.split('_',expand=True)[0]
            self.df_classify['Class'] = self.df_classify['Image_Label'].str.split('_',expand=True)[1]
        else:
            self.df_classify = None
        self.segment = None

    def process(self,x,ids):
        
        y = []
        labels = []
        if self.params['inference_model']['classify_path'] != None:
            for i in ids:
                i = i.split('/')[-1]
                i = self.df_classify[self.df_classify['Image'] == i]
                y_tp = []
                x_tp = []
                for it,val in zip(i['EncodedPixels'],i['Class']):
                    if it == '1 1':
                        y_tp.append('1 1')
                        x_tp.append(val)
                    else:
                        y_tp.append('')
                        x_tp.append(None)
                labels.append(x_tp)
                y.append(y_tp)
            return labels,y,None,0
        else:
            if self.activation:
                x = self.activation(x)
            x = x.cpu().numpy()
            for b in range(x.shape[0]):
                pred = np.where(x[b]>self.threshold,self.names,None)
                labels.append(pred)
                y_tp = []
                for p in pred:
                    if p != None:
                        y_tp.append('1 1')
                    else:
                        y_tp.append('')
                y.append(y_tp)
        return labels,y,None,0

    def __run_check_process__(self):
        x = torch.tensor([[0.6,0.4,0.5,0.3],[0.2,0.8,0.12,0.51]])
        y = self.process(x)
        self.logger.info(f'Labels shape = {len(y)}')
        self.logger.info(f'Labels = {y}')
        
        
           