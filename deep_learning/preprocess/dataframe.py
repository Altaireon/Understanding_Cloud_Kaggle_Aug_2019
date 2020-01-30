from deep_learning.lib.utility import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Dataframe:
    def __init__(self,params):
        super(Dataframe,self).__init__()

        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.preprocess.df')
        self.params = params
        self.train = None
        self.folds = []
        self.valid = None
        self.test = None
    
    def __common__(self,df):
        tp = np.concatenate(df['Image_Label'].str.split('_').values)
        tp = np.reshape(tp,(tp.shape[0]//2,2))
        df['Class'] = tp[:,1]
        df['Class'] = df['Class'].astype('str')
        df['EncodedPixels'].fillna('',inplace=True)
        df['Label'] = np.array((df['EncodedPixels']!=''),dtype=int)
        df['ImageID'] = tp[:,0]
        if self.params['head'] == None and self.params['tail'] == None:
            return df
        elif self.params['tail'] == None:
            return df[self.params['head']:]
        else:
            return df[:self.params['tail']]
            
    
    def __process_train__(self):
        self.train = pd.read_csv(DATA_DIR+self.params['train_csv'])
        self.train = self.__common__(self.train)
        
    def __process_test__(self):
        self.test = pd.read_csv(DATA_DIR+self.params['test_csv'])
        self.test = self.__common__(self.test)
        
    def __process_valid__(self):
        if self.params['valid_size']>0:
            ids = np.unique(self.train['ImageID'].values)
            train_ids,valid_ids = train_test_split(ids,test_size=self.params['valid_size'], random_state=self.params['seed'])
            train_copy = self.train.copy()
            self.train = train_copy[train_copy['ImageID'].isin(train_ids)].copy()
            self.valid = train_copy[train_copy['ImageID'].isin(valid_ids)].copy()
            del train_copy
    
    def __process_folds__(self):
        
        self.logger.debug(f'Folds = {self.params["kfold"]}')
        if self.params['kfold']:
            skf = KFold(self.params['kfold'],random_state=self.params['seed'])
            ids = np.unique(self.train['ImageID'].values)
            for train_ids, valid_ids in skf.split(ids):
                train_copy = self.train[self.train['ImageID'].isin(ids[train_ids])].copy()
                valid_copy = self.train[self.train['ImageID'].isin(ids[valid_ids])].copy()
                self.folds.append((train_copy,valid_copy))
                del train_copy
    
    def process_dataframe(self):
        self.__process_train__()
        self.__process_test__()
        self.__process_valid__()
        self.__process_folds__()
    