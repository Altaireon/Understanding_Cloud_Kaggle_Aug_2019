from deep_learning.lib.utility import *
from deep_learning.lib.custom_optimizers import RAdam
import segmentation_models_pytorch as smp
import pretrainedmodels
from external_libs.synchronized_BatchNorm.sync_batchnorm import convert_model,SynchronizedBatchNorm2d
from deep_learning.lib.custom_train import TrainEpoch,ValidEpoch

class Train:
    def __init__(self,loader,params):
        super(Train,self).__init__()
    
        global DATA_DIR
        global LOG_DIR
        DATA_DIR = params['DATA_DIR']
        LOG_DIR = params['LOG_DIR']
        self.logger = logging.getLogger(params['logger_name']+'.train')
        self.loader = loader
        self.params = params
    
    def __get_model__(self):
        if self.params['model']['model_path'] != None:
            if self.params['device'] == 'cpu':
                self.model = torch.load(self.params['model']['model_path'], map_location=lambda storage, loc: storage)
            else:
                self.model = torch.load(self.params['model']['model_path'])
        else:
            self.model = smp.FPN(encoder_name=self.params['model']['encoder'],encoder_weights=self.params['model']['encoder_weights'],classes=self.params['num_class'],activation=self.params['model']['activation'])
        self.model = convert_model(self.model)
    
    def __get_classification_model__(self):
        if self.params['model']['model_path'] != None:
            if self.params['device'] == 'cpu':
                self.model = torch.load(self.params['model']['model_path'], map_location=lambda storage, loc: storage)
            else:
                self.model = torch.load(self.params['model']['model_path'])
        else:
            self.model = models.resnet34(num_classes=self.params['num_class'])
        self.model = convert_model(self.model)
    
    def run_check_net(self):
        pass
    
    def process_train_classification(self):
        
        self.__clean__()
        self.__get_classification_model__()
        loss = nn.BCEWithLogitsLoss()
        metrics = [accuracy_score,f1_score]
        if self.params['fold_id'] > 0:
            savePath = LOG_DIR+'model-'+self.params['id']+'/checkpoint/fold'+str(self.params['fold_id']-1)+'/'
        else:
            savePath = LOG_DIR+'model-'+self.params['id']+'/checkpoint/fold0/'
        os.mkdir(savePath)
        en_params = self.params['model']['en_params']
        en_lr = iter(en_params['lr'].keys())
        en_ep = iter(en_params['lr'].values())
        optimizer = RAdam([
            {'params': self.model.parameters(), 'lr': float(next(en_lr))},  
        ])
        train_epoch = TrainEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            is_classify=True,
            params=self.params,
            device=self.params['device'],
            verbose=True,
        )
        
        valid_epoch = ValidEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics,
            is_classify=True,
            params=self.params,
            device=self.params['device'],
            verbose=True,
        )

        max_score = 0
        
        en_iter = next(en_ep)
        self.logger.info(savePath)
        r = 0
        for i in range(0, self.params['num_epochs']):
            if self.params['model']['early_stopping'] != -1 and self.params['model']['early_stopping'] <= r:
                break
            if i <= self.params['start_epoch']:
                continue
            self.logger.info('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(self.loader.train_loader)
            valid_logs = valid_epoch.run(self.loader.valid_loader)
            self.logger.info(train_logs)
            self.logger.info(valid_logs)
            if max_score < valid_logs['f1_score']:
                max_score = valid_logs['f1_score']
                torch.save(self.model, f'{savePath}{i}_{max_score}.pth')
                self.logger.info('Model saved!')
                r = 0
            else:
                if not self.params['inference_model']['keep_best']:
                    torch.save(self.model, f'{savePath}{i}_{max_score}.pth')
                r = r + 1
            if i == en_iter-1:
                try:
                    x=float(next(en_lr))
                    print()
                    optimizer.param_groups[0]['lr'] = x
                    self.logger.info(f'Decrease encoder learning rate to {x}')
                    en_iter = en_iter + next(en_ep)
                except:
                    break
        if self.params['valid_size'] > 0:
            valid_logs = valid_epoch.run(self.loader.valid_loader)
            self.logger.info(valid_logs)
    
    def process_kfold_train_classification(self):
        self.__clean__()
        
        if self.params['fold_id'] > 0:
            self.logger.info(f'Fold = {self.params["fold_id"]}')
            self.loader.train_loader,self.loader.valid_loader = self.loader.kfold_loader[self.params['fold_id']-1]
            self.process_train_classification()
            
        else:
            for fold,(train_loader,valid_loader) in enumerate(self.loader.kfold_loader):
                self.logger.info(f'Fold = {fold+1}')
                self.__get_classification_model__()
                loss = nn.BCEWithLogitsLoss()
                metrics = [accuracy_score,f1_score]
                savePath = LOG_DIR+'model-'+self.params['id']+'/checkpoint/fold'+str(fold)+'/'
                os.mkdir(savePath)
                en_params = self.params['model']['en_params']
                en_lr = iter(en_params['lr'].keys())
                en_ep = iter(en_params['lr'].values())
                optimizer = RAdam([
                    {'params': self.model.parameters(), 'lr': float(next(en_lr))},  
                ])
                train_epoch = TrainEpoch(
                    self.model, 
                    loss=loss, 
                    metrics=metrics, 
                    optimizer=optimizer,
                    is_classify=True,
                    params=self.params,
                    device=self.params['device'],
                    verbose=True,
                )
                
                valid_epoch = ValidEpoch(
                    self.model, 
                    loss=loss, 
                    metrics=metrics,
                    is_classify=True,
                    params=self.params,
                    device=self.params['device'],
                    verbose=True,
                )
        
                max_score = 0
                
                en_iter = next(en_ep)
                self.logger.info(savePath)
                r = 0
                for i in range(0, self.params['num_epochs']):
                    if self.params['model']['early_stopping'] != -1 and self.params['model']['early_stopping'] <= r:
                        break
                    if i <= self.params['start_epoch']:
                        continue
                    self.logger.info('\nEpoch: {}'.format(i))
                    train_logs = train_epoch.run(train_loader)
                    valid_logs = valid_epoch.run(valid_loader)
                    self.logger.info(train_logs)
                    self.logger.info(valid_logs)
                    if max_score < valid_logs['f1_score']:
                        max_score = valid_logs['f1_score']
                        torch.save(self.model, f'{savePath}{i}_{max_score}.pth')
                        self.logger.info('Model saved!')
                        r = 0
                    else:
                        r = r + 1
                    if i == en_iter-1:
                        try:
                            x=float(next(en_lr))
                            print()
                            optimizer.param_groups[0]['lr'] = x
                            self.logger.info(f'Decrease encoder learning rate to {x}')
                            en_iter = en_iter + next(en_ep)
                        except:
                            break
                if self.params['valid_size'] > 0:
                    valid_logs = valid_epoch.run(self.loader.valid_loader)
                    self.logger.info(valid_logs)
    
    def __clean__(self):
        out_dir = LOG_DIR + 'model-' + self.params['id'] + '/'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        self.logger.debug(f'{out_dir} created..')
        
        model_dir = out_dir + 'checkpoint/'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        self.logger.debug(f'{model_dir} created..')
    
    def process_train_segmentation(self):
        
        self.__clean__()
        self.__get_model__()
        loss = smp.utils.losses.BCEDiceLoss(eps=1.)
        metrics = [
            smp.utils.metrics.IoUMetric(eps=1.),
            smp.utils.metrics.FscoreMetric(eps=1.),
        ]
        if self.params['fold_id'] > 0:
            savePath = LOG_DIR+'model-'+self.params['id']+'/checkpoint/fold'+str(self.params['fold_id']-1)+'/'
        else:
            savePath = LOG_DIR+'model-'+self.params['id']+'/checkpoint/fold0/'
        os.mkdir(savePath)
        en_params = self.params['model']['en_params']
        de_params = self.params['model']['de_params']
        en_lr = iter(en_params['lr'].keys())
        en_ep = iter(en_params['lr'].values())
        de_lr = iter(de_params['lr'].keys())
        de_ep = iter(de_params['lr'].values())
        optimizer = RAdam([
            {'params': self.model.decoder.parameters(), 'lr': float(next(de_lr))}, 
            {'params': self.model.encoder.parameters(), 'lr': float(next(en_lr))},  
        ])
        train_epoch = TrainEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            params=self.params,
            device=self.params['device'],
            verbose=True,
        )
        
        valid_epoch = ValidEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics,
            params=self.params,
            device=self.params['device'],
            verbose=True,
        )

        max_score = 0
        
        en_iter = next(en_ep)
        de_iter = next(de_ep)
        self.logger.info(savePath)
        r = 0
        for i in range(0, self.params['num_epochs']):
            if self.params['model']['early_stopping'] != -1 and self.params['model']['early_stopping'] <= r:
                break
            if i <= self.params['start_epoch']:
                continue
            self.logger.info('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(self.loader.train_loader)
            valid_logs = valid_epoch.run(self.loader.valid_loader)
            self.logger.info(train_logs)
            self.logger.info(valid_logs)
            if max_score < valid_logs['f-score']:
                max_score = valid_logs['f-score']
                torch.save(self.model, f'{savePath}{i}_{max_score}.pth')
                self.logger.info('Model saved!')
                r = 0
            else:
                r = r + 1
            if i == de_iter-1:
                try:
                    x=float(next(de_lr))
                    optimizer.param_groups[0]['lr'] = x
                    self.logger.info(f'Decrease decoder learning rate to {x}')
                    de_iter = de_iter + next(de_ep)
                except:
                    break
                
            if i == en_iter-1:
                try:
                    x=float(next(en_lr))
                    optimizer.param_groups[1]['lr'] = x
                    self.logger.info(f'Decrease encoder learning rate to {x}')
                    en_iter = en_iter + next(en_ep)
                except:
                    break
        if self.params['valid_size'] > 0:
            valid_logs = valid_epoch.run(self.loader.valid_loader)
            self.logger.info(valid_logs)
    
    def process_kfold_train_segmentation(self):
        
        self.__clean__()
        
        if self.params['fold_id'] > 0:
            self.logger.info(f'Fold = {self.params["fold_id"]}')
            self.loader.train_loader,self.loader.valid_loader = self.loader.kfold_loader[self.params['fold_id']-1]
            self.process_train_segmentation()
            
        else:
            for fold,(train_loader,valid_loader) in enumerate(self.loader.kfold_loader):
                self.logger.info(f'Fold = {fold+1}')
                self.__get_model__()
                loss = smp.utils.losses.BCEDiceLoss(eps=1.)
                metrics = [
                    smp.utils.metrics.IoUMetric(eps=1.),
                    smp.utils.metrics.FscoreMetric(eps=1.),
                ]
                savePath = LOG_DIR+'model-'+self.params['id']+'/checkpoint/fold'+str(fold)+'/'
                os.mkdir(savePath)
                en_params = self.params['model']['en_params']
                de_params = self.params['model']['de_params']
                en_lr = iter(en_params['lr'].keys())
                en_ep = iter(en_params['lr'].values())
                de_lr = iter(de_params['lr'].keys())
                de_ep = iter(de_params['lr'].values())
                optimizer = RAdam([
                    {'params': self.model.decoder.parameters(), 'lr': float(next(de_lr))}, 
                    {'params': self.model.encoder.parameters(), 'lr': float(next(en_lr))},  
                ])
                train_epoch = TrainEpoch(
                    self.model, 
                    loss=loss, 
                    metrics=metrics, 
                    optimizer=optimizer,
                    params=self.params,
                    device=self.params['device'],
                    verbose=True,
                )
                
                valid_epoch = ValidEpoch(
                    self.model, 
                    loss=loss, 
                    metrics=metrics,
                    params=self.params,
                    device=self.params['device'],
                    verbose=True,
                )
        
                max_score = 0
                
                en_iter = next(en_ep)
                de_iter = next(de_ep)
                self.logger.info(savePath)
                r = 0
                for i in range(0, self.params['num_epochs']):
                    if self.params['model']['early_stopping'] != -1 and self.params['model']['early_stopping'] <= r:
                        break
                    if i <= self.params['start_epoch']:
                        continue
                    self.logger.info('\nEpoch: {}'.format(i))
                    train_logs = train_epoch.run(train_loader)
                    valid_logs = valid_epoch.run(valid_loader)
                    self.logger.info(train_logs)
                    self.logger.info(valid_logs)
                    if max_score < valid_logs['f-score']:
                        max_score = valid_logs['f-score']
                        torch.save(self.model, f'{savePath}{i}_{max_score}.pth')
                        self.logger.info('Model saved!')
                        r = 0
                    else:
                        r = r + 1
                    if i == de_iter-1:
                        try:
                            x=float(next(de_lr))
                            optimizer.param_groups[0]['lr'] = x
                            self.logger.info(f'Decrease decoder learning rate to {x}')
                            de_iter = de_iter + next(de_ep)
                        except:
                            break
                        
                    if i == en_iter-1:
                        try:
                            x=float(next(en_lr))
                            optimizer.param_groups[1]['lr'] = x
                            self.logger.info(f'Decrease encoder learning rate to {x}')
                            en_iter = en_iter + next(en_ep)
                        except:
                            break
                if self.params['valid_size'] > 0:
                    valid_logs = valid_epoch.run(self.loader.valid_loader)
                    self.logger.info(valid_logs)