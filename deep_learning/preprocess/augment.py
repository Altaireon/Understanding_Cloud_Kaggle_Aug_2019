from deep_learning.lib.utility import *

class Augment:
    def __init__(self,params):
        super(Augment,self).__init__()
        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.preprocess.augment')
        self.params = params
        self.H = self.params['preprocess']['input_size'][0]
        self.W = self.params['preprocess']['input_size'][1]
        self.CH = self.params['preprocess']['crop_size'][0]
        self.CW = self.params['preprocess']['crop_size'][1]
        self.TH = self.params['preprocess']['test_size'][0]
        self.TW = self.params['preprocess']['test_size'][1]
        self.train_augment = None
        self.test_augment = None
        self.valid_augment = None
        self.tta = None
        
    def __get_training_augment__(self):
        train_transform = [
            albu.OneOf([albu.HorizontalFlip(),
                        albu.VerticalFlip()]),
            albu.ShiftScaleRotate(scale_limit=0.05,shift_limit=0.05,rotate_limit=180,p=0.4),
            albu.OneOf([albu.OpticalDistortion(p=1/4),albu.GridDistortion(p=3/4)],p=0.35),
            albu.OneOf([albu.RandomBrightness(p=2/5),albu.RandomContrast(p=2/5),albu.IAASharpen(p=1/5)],p=0.4),
            albu.Resize(self.H,self.W),
#            albu.OneOf([albu.RandomCrop(self.CH,self.CW,p=0.5),albu.CropNonEmptyMaskIfExists(self.CH,self.CW,p=0.5)],p=1.0),
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Training Augumentation Loaded')
        return albu.Compose(train_transform)
    
    def __get_validation_augment__(self):
        test_transform = [
            albu.Resize(self.H,self.W),
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Validation Augumentation Loaded')
        return albu.Compose(test_transform)
    
    def __get_testing_augment__(self):
        
        tta = [
            albu.HorizontalFlip(p=1.0),
            albu.VerticalFlip(p=1.0),
            albu.RandomContrast(limit=0.1,p=1.0),
            albu.RandomBrightness(limit=0.1,p=1.0)
        ]
        test_transform = [
            albu.Resize(self.TH,self.TW),
            albu.Normalize(),
            ToTensorV2()
        ]
        tfs = [albu.Compose(test_transform)]
        for t in tta:
            t = [t]
            t.extend(test_transform)
            tf = albu.Compose(t)
            tfs.append(tf)
        t = [None]
        t.extend(tta)
        self.logger.debug('Testing Augumentation Loaded')
        self.logger.debug(f'Size of TTA {len(t)}')
        return tfs,t
    
    def process_augment(self):
        self.train_augment = self.__get_training_augment__()
        self.valid_augment = self.__get_validation_augment__()
        self.test_augment,self.tta  = self.__get_testing_augment__()
    
    def run_check_augument(self):
        img = cv2.imread(DATA_DIR+'train_images/0a7a247.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        aug = self.__get_training_augment__()
        mask = np.zeros(img.shape)
        mask[100:200,350:450] = 1
        img = aug(image=img,mask=mask)['image']
        img = np.transpose(img,(1,2,0))
        plt.title('Train Augument')
        plt.imshow(img,)
        plt.show()
        
        img = cv2.imread(DATA_DIR+'train_images/0a7a247.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.__get_validation_augment__()
        mask = np.zeros(img.shape)
        mask[100:200,350:450] = 1
        img = aug(image=img,mask=mask)['image']
        img = np.transpose(img,(1,2,0))
        plt.title('Valid Augument')
        plt.imshow(img)
        plt.show()
        
        img = cv2.imread(DATA_DIR+'train_images/0a7a247.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.__get_testing_augment__()
        mask = np.zeros(img.shape)
        mask[100:200,350:450] = 1
        img = aug(image=img,mask=mask)['image']
        img = np.transpose(img,(1,2,0))
        plt.title('Test Augument')
        plt.imshow(img)
        plt.show()
        
        