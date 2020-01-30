from data.loader import *
from deep_learning.preprocess.preprocess import *
from deep_learning.train.train import *
from deep_learning.test.test import *
from deep_learning.preprocess.visualize import *
from deep_learning.postprocess.postprocess import *
import logging
import json
import pretrainedmodels

with open('params/cloud-1.json') as f:
    params = json.loads(f.read())
    f.close()
    
logfile = LOG_DIR + params['name'] + '-' + params['id'] + '.log'

logger = logging.getLogger(params['logger_name'])
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(logfile,'w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info("Starting Pipeline...")

pp = PreProcess(params)
pp.process_dataframe()
#pp.run_check_augument()

loader = Loader(pp,params)
loader.process_data()
#loader.run_check_loader()

viz = Visualize(params)
#viz.visualize_images()
##
#train = Train(loader,params)
#train.process_kfold_train_segmentation()

pop = PostProcess(params,rle_encoding,loader.names)
#pop.run_check_postprocess()

test = Test(loader,pop,params)
test.process_test_segmentation()

viz.visualize_test_images()

logger.info("End Pipeline")

logging.shutdown()
