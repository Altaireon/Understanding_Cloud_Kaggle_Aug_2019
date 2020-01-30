from deep_learning.lib.utility import *
from deep_learning.lib.custom_optimizers import RAdam
import segmentation_models_pytorch as smp
from external_libs.synchronized_BatchNorm.sync_batchnorm import convert_model
from deep_learning.postprocess.common import sharpen
# Design Code to get rid of duplicates

class Test:
    def __init__(self,loader,postprocess,params):
        super(Test,self).__init__()
        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.test')
        self.loader = loader
        self.params = params
        self.postprocess = postprocess
        self.model = None
        
    def __get_model__(self,weigth_path=None):
        if weigth_path:
            if self.params['device'] == 'cpu':
                self.model = torch.load(weigth_path, map_location=lambda storage, loc: storage)
            else:
                self.model = torch.load(weigth_path)
            self.model.eval()
            self.model.to(self.params['device'])
    
    def __update_path__(self,path,crit,th):
        m = -1
        out = None
        for file in glob.glob(path+'*'):
            value = float(file.split('/')[-1].rsplit('.',1)[0].split('_')[crit-1])
            if th == None and value > m:
                m = value
                out = file
            elif type(th) != str and value == th:
                return file
        return out
    
    def process_test_segmentation(self): # i
        ensemble_type = self.params['inference_model']['ensemble_type']
        if ensemble_type == None:
            for i,model in enumerate(self.params['inference_model']['models']):
                savePath = LOG_DIR+'model-' + model + '/checkpoint/'
                output_path = LOG_DIR+'model-' + model + '/output/'
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                os.mkdir(output_path)
                fold_paths = self.params['inference_model']['path'][i]
                fold_criterions = self.params['inference_model']['criterion'][i]
                fold_thresholds = self.params['inference_model']['threshold'][i]
                typ = self.params['inference_model']['type'][i]
                tta_typ = self.params['inference_model']['tta_type'][i]
                crit = self.params['inference_model']['tta_criterion'][i]
                self.logger.debug(fold_paths)
                self.logger.debug(fold_criterions)
                self.logger.debug(fold_thresholds)
                self.logger.debug(typ)
                if typ == None:
                    self.__seperate_prediction__(fold_paths,fold_criterions,fold_thresholds,savePath,output_path,tta_typ)
                else:
                    self.__agg_prediction__(fold_paths,fold_criterions,fold_thresholds,savePath,output_path,crit,tta_typ,typ)
        else:
            if self.params['inference_model']['preds_path'] == None:
                dfs = []
                for i,model in enumerate(self.params['inference_model']['models']):
                    savePath = LOG_DIR+'model-' + model + '/checkpoint/'
                    output_path = LOG_DIR+'model-' + model + '/output/'
                    if os.path.exists(output_path):
                        shutil.rmtree(output_path)
                    os.mkdir(output_path)
                    fold_paths = self.params['inference_model']['path'][i]
                    fold_criterions = self.params['inference_model']['criterion'][i]
                    fold_thresholds = self.params['inference_model']['threshold'][i]
                    typ = self.params['inference_model']['type'][i]
                    tta_typ = self.params['inference_model']['tta_type'][i]
                    crit = self.params['inference_model']['tta_criterion'][i]
                    self.logger.debug(fold_paths)
                    self.logger.debug(fold_criterions)
                    self.logger.debug(fold_thresholds)
                    self.logger.debug(typ)
                    if typ == None:
                        dfs.extend(self.__seperate_prediction__(fold_paths,fold_criterions,fold_thresholds,savePath,output_path,tta_typ))
                    else:
                        dfs.append(self.__agg_prediction__(fold_paths,fold_criterions,fold_thresholds,savePath,output_path,crit,tta_typ,typ))
                df = dfs[0]
                for i in range(1,len(dfs)):
                    df = pd.merge(df,dfs[i],on='Image_Label')
                    
                df.fillna('',axis=1,inplace=True)
                ids = df['Image_Label']
                preds = []
                for col in df[df.columns[df.columns != 'Image_Label']]:
                    preds.append(df[col].values)
                self.logger.info(df['Image_Label'].head())
                self.logger.info(len(preds))
                self.__ensemble_prediction__(ids,preds,self.params['inference_model']['ensemble_type'])
                
            else:
                dfs = []
                for p in self.params['inference_model']['preds_path']:
                    p = LOG_DIR + p
                    dfs.append(pd.read_csv(p))
                    
                df = dfs[0]
                for i in range(1,len(dfs)):
                    df = pd.merge(df,dfs[i],on='Image_Label',suffixes=("_x"+str(i),"_y"+str(i)))
                df.fillna('',axis=1,inplace=True)
                ids = df['Image_Label']
                preds = []
                for col in df.columns[df.columns != 'Image_Label']:
                    preds.append(df[col].values)
                self.logger.info(df['Image_Label'].head())
                self.logger.info(len(preds))
                self.logger.info(len(ids))
                self.__ensemble_prediction__(ids,preds,self.params['inference_model']['ensemble_type'])
            
                
    def __seperate_prediction__(self,fold_paths,fold_criterions,fold_thresholds,savePath,output_path,tta_typ):
        
        dfs = []
        for i,path in enumerate(fold_paths):#self.params['inference_model']['path']
            weight_path = savePath + path + '/'
            fold_path = output_path + path + '.csv'
            criterion = fold_criterions[i] #self.params['inference_model']['criterion']
            threshold = fold_thresholds[i]
            weight_path = self.__update_path__(weight_path,criterion,threshold)
            self.logger.info(f'loading {weight_path} ...')
            self.__get_model__(weight_path)
            itrs = []
            pred = []
            ids = []
            for i in self.loader.test_loader:
                itrs.append(iter(i))
            counter = 0
            for _ in tqdm.tqdm(range(len(self.loader.test_loader[0]))):
                x = []
                for itr in itrs:
                    x_tp,_,_,image_id = next(itr)
                    x.append(x_tp.to(self.params['device']))
                tp_ids = []
                for i in image_id:
                    tp_ids.append(DATA_DIR+self.params['path']['test']+i)
                y_pred = self.__tta_batch_update__(x,tta_typ,tp_ids)
                if self.params['inference_model']['classify_path'] != None or  self.params['inference_model']['segment_path'] != None:
                    labels,masks,_,c = self.postprocess.process_all(y_pred,tp_ids)
                elif self.params['is_classify']:
                    labels,masks,_,c = self.postprocess.process_classify(y_pred,tp_ids)
                else:
                    labels,masks,_,c = self.postprocess.process_segment(y_pred,tp_ids)
                counter = counter + c
                for itr_batch,(label,mask) in enumerate(zip(labels,masks)):
                    for itr_class in range(self.params['num_class']):
                        if label[itr_class] != None and mask[itr_class] != None:
                            pred.append(mask[itr_class])
                        else:
                            pred.append('')
                        ids.append(image_id[itr_batch]+'_'+self.loader.names[itr_class])
#                 pred.extend(y_pred)
#                 ids.extend(image_id)
            self.logger.info(f'Adaptive Thresholding Count = {counter}')
            output = {'Image_Label':ids,'EncodedPixels':pred}
            df = pd.DataFrame(output)
            df.to_csv(fold_path,index=False)
            dfs.append(df)
        return dfs
                
            
    def __agg_prediction__(self,fold_paths,fold_criterions,fold_thresholds,savePath,output_path,crit,tta_typ,typ):
        
        out_path = output_path + 'output.csv'
        my_models = []
        for i,path in enumerate(fold_paths):
            weight_path = savePath + path + '/'
            criterion = fold_criterions[i]
            threshold = fold_thresholds[i]
            weight_path = self.__update_path__(weight_path,criterion,threshold)
            self.logger.info(f'loading {weight_path} ...')
            self.__get_model__(weight_path)
            my_models.append(self.model)
        itrs = []
        pred = []
        ids = []
        counter = 0
        for i in self.loader.test_loader:
            itrs.append(iter(i))
        for _ in tqdm.tqdm(range(len(self.loader.test_loader[0]))):
            self.model= my_models
            x = []
            for itr in itrs:
                x_tp,_,_,image_id = next(itr)
                x.append(x_tp.to(self.params['device']))
            y_pred = self.__tta_batch_agg_update__(x,crit,tta_typ,typ)
            tp_ids = []
            for i in image_id:
                tp_ids.append(DATA_DIR+self.params['path']['test']+i)
            if self.params['inference_model']['classify_path'] != None or  self.params['inference_model']['segment_path'] != None:
                labels,masks,_,c = self.postprocess.process_all(y_pred,tp_ids)
            elif self.params['is_classify']:
                labels,masks,_,c = self.postprocess.process_classify(y_pred,tp_ids)
            else:
                labels,masks,_,c = self.postprocess.process_segment(y_pred,tp_ids)
            counter = counter + c
            for itr_batch,(label,mask) in enumerate(zip(labels,masks)):
                for itr_class in range(self.params['num_class']):
                    if label[itr_class] != None and mask[itr_class] != None:
                        pred.append(mask[itr_class])
                    else:
                        pred.append('')
                    ids.append(image_id[itr_batch]+'_'+self.loader.names[itr_class])
#                    pred.extend(y_pred)
#                    ids.extend(image_id)
        self.logger.info(f'Adaptive Thresholding Count = {counter}')
        output = {'Image_Label':ids,'EncodedPixels':pred}
        df = pd.DataFrame(output)
        df.to_csv(out_path,index=False)
        return df
            
    def __ensemble_prediction__(self,ids,preds,typ):
        
        output_path = LOG_DIR + 'ensemble/'
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        output_path =  output_path + 'output.csv'
        y = []
        for i in tqdm.tqdm(range(len(ids))):
            y_tp = []
            for p in preds:
                if not self.params['is_classify']:
                    d = self.postprocess.decoding(p[i],self.params['inference_model']['mask_size'][0],self.params['inference_model']['mask_size'][1])
                else:
                    d = self.postprocess.decoding(p[i],1,1)
                y_tp.append(d)
            if typ == "union":
                y_tp = np.stack(y_tp)
                y_tp = np.max(y_tp,axis=0)
            elif typ == "intersection":
                y_tp = np.stack(y_tp)
                y_tp = np.min(y_tp,axis=0)
            elif typ == "voting":
                y_tp = np.stack(y_tp)
                s = y_tp.shape[0]
                y_tp = np.sum(y_tp,axis=0)/s
                y_tp[y_tp>self.params['inference_model']['voting_threshold']] = 1
                y_tp[y_tp<=self.params['inference_model']['voting_threshold']] = 0
            y.append(self.postprocess.encoding(y_tp))
        output = {'Image_Label':ids,'EncodedPixels':y}
        df = pd.DataFrame(output)
        df.to_csv(output_path,index=False)
        
    def __tta_batch_agg_update__(self,xs,crit,tta_typ,typ):
        
        if crit == 1:
            y = []
            for i,x in enumerate(xs):
                pred = self.__batch_agg_update__(x,typ)
                if not self.params['is_classify']:
                    pred = inverse_tta(pred,self.loader.preprocess.aug.tta[i])
                if self.params['inference_model']['sharpen'] and i > 0:
                    pred = sharpen(pred)
                y.append(pred)
            y = torch.stack(y)
            if tta_typ == "max":
                y,_ = torch.max(y,dim=0)
            elif tta_typ == "min":
                y,_ = torch.min(y,dim=0)
            elif tta_typ == "mean":
                y = torch.mean(y,dim=0)
            return y
        
        elif crit == 2:
            y = []
            my_models = self.model
            for m in my_models:
                self.model = m  
                y_tp = []    
                for i,x in enumerate(xs):
                    pred = self.__batch_update__(x)
                    if not self.params['is_classify']:
                        pred = inverse_tta(pred,self.loader.preprocess.aug.tta[i])
                    if self.params['inference_model']['sharpen'] and i > 0:
                        pred = sharpen(pred)
                    y_tp.append(pred)
                y_tp = torch.stack(y_tp)
                if tta_typ == "max":
                    y_tp,_ = torch.max(y_tp,dim=0)
                elif tta_typ == "min":
                    y_tp,_ = torch.min(y_tp,dim=0)
                elif tta_typ == "mean":
                    y_tp = torch.mean(y_tp,dim=0)
                y.append(y_tp)
            y = torch.stack(y)
            if typ == "max":
                y,_ = torch.max(y,dim=0)
            elif typ == "min":
                y,_ = torch.min(y,dim=0)
            elif typ =="mean":
                y = torch.mean(y,dim=0)     
#            elif typ == "voting":
#                pred = []
#                for p in y:
#                    pred.append(self.postprocess.process_segment(p,ids))
#                pred = np.stack(pred)
#                pred = np.sum(pred,axis=0)/2
#                pred[pred>self.params['inference_model']['tta_voting_threshold']] = 1
#                pred[pred<=self.params['inference_model']['tta_voting_threshold']] = 0
            return y
        
    def __tta_batch_update__(self,xs,tta_typ,ids):
        
        y = []
        for i,x in enumerate(xs):
            pred = self.__batch_update__(x)
            if not self.params['is_classify']:
                pred = inverse_tta(pred,self.loader.preprocess.aug.tta[i])
            if self.params['inference_model']['sharpen'] and i > 0:
                pred = sharpen(pred)
            y.append(pred)
        y = torch.stack(y)
        if tta_typ == "max":
            y,_ = torch.max(y,dim=0)
        elif tta_typ == "min":
            y,_ = torch.min(y,dim=0)
        elif tta_typ == "mean":
            y = torch.mean(y,dim=0)
#        elif tta_typ == "voting":
#            pred = []
#            for p in y:
#                pred.append(self.postprocess.process_segment(p,ids))
#            pred = np.stack(pred)
#            pred = np.sum(pred,axis=0)/2
#            pred[pred>self.params['inference_model']['tta_voting_threshold']] = 1
#            pred[pred<=self.params['inference_model']['tta_voting_threshold']] = 0
        return y
        
    def __batch_update__(self, x):
        self.model.eval()
        with torch.no_grad():
            if self.params['parallelize']:
                prediction = data_parallel(self.model,x)
            else:
                prediction = self.model(x)
        return prediction
    
    def __batch_agg_update__(self,x,typ,ids):
        pred = []
        for m in self.model:
            m.eval()
            with torch.no_grad():
                if self.params['parallelize']:
                    pred.append(data_parallel(m,x))
                else:
                    pred.append(m(x))
        pred = torch.stack(pred)
        if typ == "max":
            pred,_ = torch.max(pred,dim=0)
        elif typ == "min":
            pred,_ = torch.min(pred,dim=0)
        elif typ == "mean":
            pred = torch.mean(pred,dim=0)
#        elif typ == "voting":
#            preds = []
#            for p in pred:
#                preds.append(predict_mask, num = thresholding_and_minsize(predict_mask,self.params['inference_model']['voting_threshold'],0,(1400,2100))
#            preds = np.stack(preds)
#            preds = np.sum(preds,axis=0)/
#            preds[preds>self.params['inference_model']['voting_threshold']] = 1
#            preds[preds<=self.params['inference_model']['voting_threshold']] = 0
#            pred = preds
        return pred