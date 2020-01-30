from deep_learning.lib.utility import *
from segmentation_models_pytorch.utils.meter import AverageValueMeter
 
class Epoch:
    
    def __init__(self, model, loss, metrics, stage_name,is_classify=False, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.is_classify = is_classify
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        if not self.is_classify:
            for metric in self.metrics:
                metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        with tqdm.tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, label, _ in iterator:
                x, y,label = x.to(self.device), y.to(self.device),label.to(self.device)
                if self.is_classify:
                    loss, y_pred = self.batch_update(x,label)
                else:
                    loss, y_pred = self.batch_update(x, y)
                
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                try:
                    loss_logs = {self.loss.__name__: loss_meter.mean}
                except:
                    loss_logs = {"loss": loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    if self.is_classify:
                        metric_value = metric_fn(y_pred, label)
                    else:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs
    

class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer,params,is_classify=False, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            is_classify=is_classify,
            device=device,
            verbose=verbose,
        )
        self.params = params
        self.optimizer = optimizer
        self.gr_accum = self.params['model']['gr_accum']
        self.optimizer.zero_grad()
        self.i = 0

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        if self.params['parallelize']:
            prediction = data_parallel(self.model,x)
        else:
            prediction = self.model(x)
        y_loss = self.loss(prediction, y)
        loss = y_loss / self.gr_accum
        loss.backward()
        self.i = self.i+1
        if (self.i % self.gr_accum) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return y_loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics,params,is_classify=False, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            is_classify=is_classify,
            device=device,
            verbose=verbose,
        )
        self.params=params

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        torch.cuda.empty_cache()
        with torch.no_grad():
            if self.params['parallelize']:
                prediction = data_parallel(self.model,x)
            else:
                prediction = self.model(x)
            loss = self.loss(prediction, y)
        return loss, prediction
    