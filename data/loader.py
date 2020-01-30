from data.lib.utility import *
import torch

class _CloudDataset(Dataset):
    def __init__(self, df, mode, logger_name, path, preprocess=None,sample_data_train=0):
        self.logger = logging.getLogger(logger_name+'.data.dataset')
        self.mode    = mode
        self.preprocess = preprocess
        self.df = df
        self.num_image = self.df.shape[0]//4
        self.path = DATA_DIR+path
        self.uid = df['ImageID'].unique()
        self.counter=0
        self.sample_data_train=sample_data_train

    def __str__(self):
        num1 = (self.df['Class']=='Fish').sum()
        num2 = (self.df['Class']=='Flower').sum()
        num3 = (self.df['Class']=='Gravel').sum()
        num4 = (self.df['Class']=='Sugar').sum()
        pos1 = ((self.df['Class']=='Fish') & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']=='Flower') & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']=='Gravel') & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']=='Sugar') & (self.df['Label']==1)).sum()
        neg1 = num1-pos1
        neg2 = num2-pos2
        neg3 = num3-pos3
        neg4 = num4-pos4

        num = self.df.shape[0]
        pos = (self.df['Label']==1).sum()
        neg = num-pos


        string  = '\n'
        string += '\tmode    = %s\n'%self.mode
        string += '\tnum_image = %8d\n'%self.num_image
        string += '\tlen       = %8d\n'%num
        if self.mode in ['train','valid']:
            string += '\t\tpos1, neg1 = %5d  %0.3f,  %5d  %0.3f\n'%(pos1,pos1/num,neg1,neg1/num)
            string += '\t\tpos2, neg2 = %5d  %0.3f,  %5d  %0.3f\n'%(pos2,pos2/num,neg2,neg2/num)
            string += '\t\tpos3, neg3 = %5d  %0.3f,  %5d  %0.3f\n'%(pos3,pos3/num,neg3,neg3/num)
            string += '\t\tpos4, neg4 = %5d  %0.3f,  %5d  %0.3f\n'%(pos4,pos4/num,neg4,neg4/num)
        return string

    def __len__(self):
        return len(self.uid)
    
    def __getitem__(self, index):
        image_id = self.uid[index-1]
        label = np.reshape(np.array([
            self.df.loc[self.df['Image_Label']==image_id + '_Fish','Label'].values[0],
            self.df.loc[self.df['Image_Label']==image_id + '_Flower','Label'].values[0],
            self.df.loc[self.df['Image_Label']==image_id + '_Gravel','Label'].values[0],
            self.df.loc[self.df['Image_Label']==image_id + '_Sugar','Label'].values[0],
        ],dtype=int),(4,1,1))
        rle = [
            self.df.loc[self.df['Image_Label']==image_id + '_Fish','EncodedPixels'].values[0],
            self.df.loc[self.df['Image_Label']==image_id + '_Flower','EncodedPixels'].values[0],
            self.df.loc[self.df['Image_Label']==image_id + '_Gravel','EncodedPixels'].values[0],
            self.df.loc[self.df['Image_Label']==image_id + '_Sugar','EncodedPixels'].values[0],
        ]
        image = cv2.imread(self.path+image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = np.array([rle_decoding(r, image.shape[0], image.shape[1], fill_value=1) for c,r in zip([0,1,2,3],rle)],dtype=int)
        if self.counter < self.sample_data_train:
            self.counter = self.counter + 1
            for idx,mask_decoded in enumerate(mask):
                img = image.copy()
                if len(np.unique(mask_decoded)) > 1:
                    mask_decoded = np.where(mask_decoded==1,255,0)
                    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
                    ax[0].imshow(img)
                    ax[1].imshow(mask_decoded)
                    plt.savefig(f'{DATA_DIR}/samples/{index}_{idx}.png')
        mask  = np.transpose(mask,(1,2,0))   
        ag = self.preprocess(image=image,mask=mask)
        image = ag['image']
        mask = ag['mask']
        mask  = mask.permute((2,0,1)).float()
        label = torch.Tensor(label).view(4)
        return image, mask, label, image_id
    
class Loader():
    def __init__(self,preprocess, params):
        super(Loader,self).__init__()

        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.data.dataloader')
        self.train_loader   = None
        self.valid_loader   = None
        self.test_loader    = None
        self.kfold_loader = []
        self.params = params
        self.preprocess = preprocess
        self.names = None
        if self.params['sample_data_train']:
            if os.path.exists(f'{DATA_DIR}samples'):
                shutil.rmtree(f'{DATA_DIR}samples')
            os.makedirs(f'{DATA_DIR}samples')
    
    def __process_train__(self):
        train_dataset = _CloudDataset(
            mode    = 'train',
            df     = self.preprocess.df.train,
            logger_name = self.params['logger_name'],
            path = self.params['path']['train'],
            preprocess = self.preprocess.aug.train_augment,
            sample_data_train=self.params['sample_data_train']
        )
        self.logger.debug(str(train_dataset))
        self.logger.debug("Setting up Train Loader => STARTING")
        train_loader = DataLoader(
                train_dataset,
    #            sampler     = ImbalancedDatasetSampler(train_dataset),
                sampler    = RandomSampler(train_dataset),
    #            sampler     = FiveBalanceClassSampler(train_dataset),
                batch_size  = self.params['batch_size'][0],
                drop_last   = True,
                num_workers = 1,
                pin_memory  = True,
    #            collate_fn  = null_collate
            )
        self.logger.debug("Setting up Train Loader => SUCCESS")
        return train_loader
    
    def __process_kfold__(self):
        if self.params['kfold']:
            kfold_loader = []
            for i,(train,test) in enumerate(self.preprocess.df.folds):
                self.logger.debug(f'fold = {i}')
                train_dataset = _CloudDataset(
                    mode    = 'train',
                    df     = train,
                    logger_name = self.params['logger_name'],
                    path = self.params['path']['train'],
                    preprocess = self.preprocess.aug.train_augment,
                    sample_data_train=self.params['sample_data_train']
                )
                self.logger.debug(str(train_dataset))
                self.logger.debug("Setting up Train Loader => STARTING")
                train_loader = DataLoader(
                    train_dataset,
        #            sampler     = ImbalancedDatasetSampler(train_dataset),
                    sampler    = RandomSampler(train_dataset),
        #            sampler     = FiveBalanceClassSampler(train_dataset),
                    batch_size  = self.params['batch_size'][0],
                    drop_last   = True,
                    num_workers = 1,
                    pin_memory  = True,
        #            collate_fn  = null_collate
                )
                self.logger.debug("Setting up Train Loader => SUCCESS")
                valid_dataset = _CloudDataset(
                    mode    = 'valid',
                    df     = test,
                    logger_name = self.params['logger_name'],
                    path = self.params['path']['valid'],
                    preprocess = self.preprocess.aug.valid_augment,
                    sample_data_train=self.params['sample_data_train']
                )
                self.logger.debug(str(valid_dataset))
                self.logger.debug("Setting up Valid Loader => STARTING")
                valid_loader = DataLoader(
                    valid_dataset,
                    sampler     = SequentialSampler(valid_dataset),
                    batch_size  = self.params['batch_size'][1],
                    drop_last   = False,
                    num_workers = 1,
                    pin_memory  = True,
        #            collate_fn  = null_collate
                )
                kfold_loader.append((train_loader,valid_loader))
                self.logger.debug("Setting up Valid Loader => SUCCESS")
            return kfold_loader
        
        return None
    
    def __process_valid__(self):
        if self.params['valid_size']>0:
            valid_dataset = _CloudDataset(
                mode    = 'valid',
                df     = self.preprocess.df.valid,
                logger_name = self.params['logger_name'],
                path = self.params['path']['valid'],
                preprocess = self.preprocess.aug.valid_augment,
                sample_data_train=self.params['sample_data_train']
            )
            self.logger.debug(str(valid_dataset))
            self.logger.debug("Setting up Valid Loader => STARTING")
            valid_loader = DataLoader(
                valid_dataset,
                sampler     = SequentialSampler(valid_dataset),
                batch_size  = self.params['batch_size'][1],
                drop_last   = False,
                num_workers = 1,
                pin_memory  = True,
    #            collate_fn  = null_collate
            )
            self.logger.debug("Setting up Valid Loader => SUCCESS")
            return valid_loader
        return None
    
    def __process_test__(self):
        test_loaders = []
        for aug in self.preprocess.aug.test_augment:
            test_dataset = _CloudDataset(
                mode    = 'test',
                df     = self.preprocess.df.test,
                logger_name = self.params['logger_name'],
                path = self.params['path']['test'],
                preprocess = aug,
                sample_data_train=self.params['sample_data_train']
            )
            self.logger.debug(str(test_dataset))
            self.logger.debug("Setting up Test Loader => STARTING")
            test_loaders.append(DataLoader(
                    test_dataset,
                    sampler    = SequentialSampler(test_dataset),
                    batch_size  = self.params['batch_size'][2],
                    drop_last   = False,
                    num_workers = 1,
                    pin_memory  = True,
        #            collate_fn  = null_collate
                )
            )
        self.logger.debug("Setting up Test Loader => SUCCESS")
        return test_loaders
    
    def process_data(self):
        self.logger.debug(f'\nBatch Size:\nTrain: {self.params["batch_size"][0]}\nValid: {self.params["batch_size"][1]}\nTest: {self.params["batch_size"][2]}')
        self.train_loader = self.__process_train__()
        self.valid_loader = self.__process_valid__()
        self.test_loader  = self.__process_test__()
        self.kfold_loader = self.__process_kfold__()
        self.names = self.train_loader.dataset.df.Class.unique()
        self.names.sort()
        self.logger.debug(f'Processing Data => SUCCESS')
        
    def run_check_loader(self):
        self.logger.info("Training Data.........")
        for img,mask,_,_ in self.train_loader:
            self.logger.info(f'Image Shape = {img.shape}')
            assert(img.shape == (self.params['batch_size'][0],3,self.params['preprocess']['crop_size'][0],self.params['preprocess']['crop_size'][1]))
            self.logger.info(f'Mask Shape = {mask.shape}')
            assert(mask.shape == (self.params['batch_size'][0],self.params['num_class'],self.params['preprocess']['crop_size'][0],self.params['preprocess']['crop_size'][1]))
            break
#        itr = 0
#        for img,_,_,_ in tqdm.tqdm(self.train_loader):
#            itr = itr + img.shape[0]
#        assert(itr == self.train_loader.dataset.df.shape[0])
        
        self.logger.info("Validation Data.........")
        for img,mask,_,_ in self.valid_loader:
            self.logger.info(f'Image Shape = {img.shape}')
            assert(img.shape == (self.params['batch_size'][1],3,self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            self.logger.info(f'Mask Shape = {mask.shape}')
            assert(mask.shape == (self.params['batch_size'][1],self.params['num_class'],self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            break
#        itr = 0
#        for img,_,_,_ in tqdm.tqdm(self.valid_loader):
#            itr = itr + img.shape[0]
#        assert(itr == self.valid_loader.dataset.df.shape[0])
        
        self.logger.info("Testing Data.........")
        for l in self.test_loader:
            for img,mask,_,_ in l:
                self.logger.info(f'Image Shape = {img.shape}')
                assert(img.shape == (self.params['batch_size'][2],3,1400,2100))
                self.logger.info(f'Mask Shape = {mask.shape}')
                assert(mask.shape == (self.params['batch_size'][2],self.params['num_class'],1400,2100))
                break
#        itr = 0
#        for img,_,_,_ in tqdm.tqdm(self.test_loader):
#            itr = itr + img.shape[0]
#        assert(itr == self.test_loader.dataset.df.shape[0])