from deep_learning.lib.utility import *

class Augment:
    def __init__(self,params):
        super(Augment,self).__init__()
        self.logger = logging.getLogger(params['logger_name']+'.preprocess.augment')
        self.params = params
        self.H = self.params['preprocess']['input_size'][0]
        self.W = self.params['preprocess']['input_size'][1]
        self.train_augment = None
        self.test_augment = None
        self.valid_augment = None
        
    def __get_training_augment__(self):
        train_transform = [
            albu.OneOf([
                    albu.HorizontalFlip(),
                    albu.VerticalFlip()],p=1.0),
            albu.OneOf([
                    albu.RandomBrightnessContrast(),
                    albu.IAASharpen(),
                    albu.CLAHE()],p=0.5),
            albu.ShiftScaleRotate(p=0.5),
            albu.RandomSnow(p=0.5),
            albu.CropNonEmptyMaskIfExists(256,400,p=1.0),
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Training Augumentation Loaded')
        return albu.Compose(train_transform)
    
    def __get_validation_augment__(self):
        test_transform = [
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Validation Augumentation Loaded')
        return albu.Compose(test_transform)
    
    def __get_testing_augment__(self):
        test_transform = [
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Testing Augumentation Loaded')
        return albu.Compose(test_transform)
    
    def process_augment(self):
        self.train_augment = self.__get_training_augment__()
        self.valid_augment = self.__get_validation_augment__()
        self.test_augment  = self.__get_testing_augment__()
        