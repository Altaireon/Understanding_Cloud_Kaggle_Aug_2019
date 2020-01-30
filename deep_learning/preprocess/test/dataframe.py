from deep_learning.lib.utility import *
from sklearn.model_selection import train_test_split

class Dataframe:
    def __init__(self,params):
        #super(Dataframe,self).__init__()
        self.logger = logging.getLogger(params['logger_name']+'.preprocess.df')
        self.params = params
        self.train = None
        self.valid = None
        self.test = None
    
    def __common__(self,df):
        tp = np.concatenate(df['Image_Label'].str.split('_').values)
        tp = np.reshape(tp,(tp.shape[0]//2,2))
        df['Class'] = tp[:,1]
        df['Class'] = df['Class'].astype('category').cat.codes
        df['EncodedPixels'].fillna('',inplace=True)
        df['Label'] = np.array((df['EncodedPixels']!=''),dtype=int)
        df['ImageID'] = tp[:,0]
        return df
    
    def __process_train__(self):
        self.train = pd.read_csv(DATA_DIR+self.params['train_csv'])
        self.train = self.__common__(self.train)
    
    def __process_test__(self):
        self.test = pd.read_csv(DATA_DIR+self.params['test_csv'])
        self.test = self.__common__(self.test)
        
    def __process_valid__(self):
        if self.params['validation']:
            ids = np.unique(self.train['ImageID'].values)
            train_ids,valid_ids = train_test_split(ids,test_size=self.params['valid_size'], random_state=self.params['seed'])
            train_copy = self.train.copy()
            self.train = train_copy[train_copy['ImageID'].isin(train_ids)].copy()
            self.valid = train_copy[train_copy['ImageID'].isin(valid_ids)].copy()
            del train_copy
    
    def process_dataframe(self):
        self.__process_train__()
        self.__process_test__()
        self.__process_valid__()