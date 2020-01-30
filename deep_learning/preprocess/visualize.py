from deep_learning.lib.utility import *
from deep_learning.preprocess.preprocess import *

class Visualize:
    def __init__(self,params):
        super(Visualize,self).__init__()

        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.params = params
        self.logger = logging.getLogger(params['logger_name']+'.visualize')
        self.params = params
        self.preprocess = PreProcess(params)
        self.preprocess.process_dataframe()
    
    def visualize_dataframe(self):
        print(self.preprocess)
    
    def visualize_images(self):
        df = self.preprocess.df.train
        imageIds = []
        for c in df.Class.unique():
            imageIds.extend(np.random.choice(np.unique(df[(df['Class']==c) & (df['Label']==1)]['ImageID'].values),self.params['visualize']['train']['sample'],replace=False))
#        outline_mask(df[df['ImageID'].isin(imageIds)].copy(),'ImageID','EncodedPixels','Class',image_path=DATA_DIR+self.params['visualize']['image_path'],save_path=DATA_DIR+self.params['visualize']['save_path'],class_names=df.Class.unique(),plot=False)
        outline_mask(df[df['ImageID'].isin(imageIds)].copy(),'ImageID','EncodedPixels','Class',image_path=DATA_DIR+self.params['visualize']['train']['image_path'],save_path=DATA_DIR+self.params['visualize']['train']['save_path'],class_names=df.Class.unique(),plot=False)
   
    def visualize_test_images(self):
        if self.params['inference_model']['ensemble_type'] == None:
            for i,model in enumerate(self.params['inference_model']['models']):
                save_path = LOG_DIR+'model-'+model+'/output/'
                self.__export_visual_data__(save_path)
        else:
            output_path = LOG_DIR + 'ensemble/'
            self.__export_visual_data__(output_path)
            
        
    def __export_visual_data__(self,save_path):
        for out in glob.glob(save_path+'*'):
            df = pd.read_csv(out)
            t_df = self.preprocess.df.test.copy()
            t_df.drop('EncodedPixels',inplace=True,axis=1)
            df = pd.merge(df,t_df)
            x = out.split('/')[-1].split('.')[0]
            output_path = save_path + x + '/'
            imageIds = []
            for c in df.Class.unique():
                imageIds.extend(np.random.choice(np.unique(df[(df['Class']==c) & (df['Label']==1)]['ImageID'].values),self.params['visualize']['test']['sample'],replace=False))
    #        outline_mask(df[df['ImageID'].isin(imageIds)].copy(),'ImageID','EncodedPixels','Class',image_path=DATA_DIR+self.params['visualize']['image_path'],save_path=DATA_DIR+self.params['visualize']['save_path'],class_names=df.Class.unique(),plot=False)
            outline_mask(df[df['ImageID'].isin(imageIds)].copy(),'ImageID','EncodedPixels','Class',image_path=DATA_DIR+self.params['visualize']['test']['image_path'],save_path=output_path,class_names=df.Class.unique(),plot=False,shape=self.params['inference_model']['mask_size'])    