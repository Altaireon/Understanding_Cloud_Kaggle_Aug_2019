from deep_learning.lib.utility import *
from deep_learning.postprocess.common import *

class Segment:
    def __init__(self,params,encoding,names,activation=None):
        
        super(Segment,self).__init__()
        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.postprocess.segment')
        self.params = params
        if type(self.params['postprocess']['threshold_pixel']) == int:
            self.thresholds = np.full((self.params['num_class'],),self.params['postprocess']['threshold_pixel'])
        else:
            self.thresholds = self.params['postprocess']['threshold_pixel']
        if type(self.params['postprocess']['min_area']) == int:
            self.min_area = np.full((self.params['num_class'],),self.params['postprocess']['min_area'])
        else:
            self.min_area = self.params['postprocess']['min_area']
        self.counter = 0
        self.activation = activation
        self.encoding=encoding
        if len(names) > 0:
            self.names=names
        else:
            self.names=range(0,self.params['num_class'])
        if self.params['inference_model']['segment_path'] != None:
            df_segment = pd.read_csv(self.params['inference_model']['segment_path'])
        else:
            df_segment = None
        self.classify = None

    def process(self,x,ids):
        if self.activation:
            x = self.activation(x)
        label = []
        mask = []
        encoded_mask = []
        z = np.zeros((self.params['inference_model']['mask_size'][0],self.params['inference_model']['mask_size'][1]))
        x = x.cpu().numpy()
        counter = 0
        for b in range(x.shape[0]):
            pred = []
            mpred = []
            lpred = []
            if type(ids[b]) == str:
                img = cv2.imread(ids[b])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img[img!=0] = 1
            img = img[:,:,0]
            c = 0
            if 'mask_size' in self.params['inference_model']:
                img = cv2.resize(img,dsize=(self.params['inference_model']['mask_size'][1],self.params['inference_model']['mask_size'][0]), interpolation=cv2.INTER_LINEAR)
            for i,threshold in enumerate(self.thresholds):
                if 'mask_size' in self.params['inference_model'] and self.params['inference_model'] != None:
                    predict_mask = cv2.resize(x[b,i],dsize=(self.params['inference_model']['mask_size'][1],self.params['inference_model']['mask_size'][0]), interpolation=cv2.INTER_LINEAR)
                predict_mask, num = thresholding_and_minsize(predict_mask,threshold,self.min_area[i],self.params['inference_model']['mask_size'])
                predict_mask = draw_convex_hull(predict_mask.astype(np.uint8),'rect')
#                self.logger.info(predict_mask.sum())
                predict_mask = np.minimum(predict_mask,img)
#                self.logger.info(predict_mask.sum())
                if num > 0 and self.classify[b][i] != None:
                    pred.append(self.encoding(predict_mask))
                    lpred.append(self.names[i])
                    mpred.append(predict_mask)
                    c = c + 1
                else:
                    pred.append(self.encoding(z))
                    lpred.append(None)
                    mpred.append(z)
            if c < self.params['inference_model']['min_count'] and self.params['inference_model']['adaptive_threshold_dec'] != None:
                pred,lpred,mpred = self.__adaptive_threshold__(x,z,b,img,self.thresholds)
                counter = counter + 1
            pred = np.stack(pred)
            lpred = np.stack(lpred)
            mpred = np.stack(mpred)
            label.append(lpred)
            encoded_mask.append(pred)
            mask.append(mpred)
        return label,encoded_mask,mask,counter
    
    def __adaptive_threshold__(self,x,z,b,img,ths):
        prev_pred = []
        prev_mpred = []
        prev_lpred = []
        for i in range(self.params['num_class']):
            prev_pred.append(self.encoding(z))
            prev_mpred.append(None)
            prev_lpred.append(z)
        dec = self.params['inference_model']['adaptive_threshold_dec']
        inc = self.params['inference_model']['adaptive_threshold_inc']
        for j in range(self.params['inference_model']['adaptive_iteration']):
            pred = []
            mpred = []
            lpred = []
            backup = ths
            tp_ths = []
            for th in ths:
                tp_ths.append(th-dec)
            ths = tp_ths 
            m_count = 100
            while True:
                if ths[0] > backup[0]:
                    break
                counter = 0
                for i,threshold in enumerate(ths):
                    if 'mask_size' in self.params['inference_model']:
                        predict_mask = cv2.resize(x[b,i],dsize=(self.params['inference_model']['mask_size'][1],self.params['inference_model']['mask_size'][0]), interpolation=cv2.INTER_LINEAR)
                    predict_mask, num = thresholding_and_minsize(predict_mask,threshold,self.min_area[i],self.params['inference_model']['mask_size'])
                    predict_mask = draw_convex_hull(predict_mask.astype(np.uint8),'rect')
                    predict_mask = np.minimum(predict_mask,img)
                    if num > 0:
                        pred.append(self.encoding(predict_mask))
                        lpred.append(self.names[i])
                        mpred.append(predict_mask)
                        counter = counter + 1 
                    else:
                        pred.append(self.encoding(z))
                        lpred.append(None)
                        mpred.append(z)
                print(counter)
                if counter >= self.params['inference_model']['adaptive_threshold_counter_min'] and counter <= self.params['inference_model']['adaptive_threshold_counter_max']:
                    return pred,lpred,mpred
                elif counter > self.params['inference_model']['adaptive_threshold_counter_max']:
                    tp_ths = []
                    backup = ths
                    for th in ths:
                        tp_ths.append(th+inc)
                    ths = tp_ths
                    if not (m_count != 100 and j==1) and counter < m_count:
                        prev_pred = pred
                        prev_mpred = mpred
                        prev_lpred = lpred
                        m_count = counter
                else:
                    break
        return prev_pred,prev_mpred,prev_lpred
               
    
    def __run_check_process__(self):
        self.logger.info(f'Min Area = {self.min_area}')
        self.logger.info(f'Thresholds = {self.thresholds}')
        x = []
        for i in range(self.params['batch_size'][2]):
            tp = []
            for _ in range(self.params['num_class']-1):
                tp.append(torch.sigmoid(torch.randn(self.params['original_size'][0],self.params['original_size'][1])))
            tp.append(torch.zeros((self.params['original_size'][0],self.params['original_size'][1])))
            tp = torch.stack(tp)
            x.append(tp)
        x = torch.stack(x)
        ly,my = self.process(x,[DATA_DIR+'lenna.bmp'])
        self.logger.info(f'Output Length = {len(ly)}')
        self.logger.info(f'Output Mask = {ly[0]}')
        self.logger.info(f'Output Length = {len(my)}')
        self.logger.info(f'Output Mask = {my[0]}')
