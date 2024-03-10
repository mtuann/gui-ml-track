import os
import json
import time
import numpy as np

# from config import get_env_config
from redis_db import RedisDB
from train_mnist import training_and_update


# CONFIG = get_env_config()
rdb = RedisDB('redis://localhost:6379', 0)

class TrainMNIST:
    def __init__(self):
        
        pass

    def run(self, hyper_params):
        pass
    

class Worker:
    def __init__(self):
        pass

    def run(self, hyper_params):
        # optimizer = hyper_params['optimizer']
        # learning_rate = hyper_params['learning_rate']
        # num_epochs = hyper_params['num_epochs']
        
        # cmd = "python train_mnist.py adam 0.001 5"
        # time.sleep(2)
        
        return {'hyper_params': hyper_params, 'results': 'done'}

worker = Worker()

import logging
logger = logging.getLogger(__name__)

def run(item):
    
    # hyper_params = {
    #     "optimizer": "adam",
    #     "learning_rate": 0.001,
    #     "num_epochs": 5
    # }
    hyper_params = item['hyper_params']

    # results = worker.run(hyper_params)
    meta_data = training_and_update(hyper_params, rdb, item)
    
    # item['results'] = results
    
    # item['status'] = 'done'
    
    # rdb.update(item)


if __name__ == '__main__':
    while True:
        item = rdb.next_request()
        if not item:
            print('Request Queue is empty!', end='\r')
            time.sleep(1)
            continue

        try:
            logger.info(f'Processing item: {item["iid"]}')
            print(f'Processing item: {item["iid"]}')
            run(item)
        except Exception as e:
            logger.info(f'Worker Error: {e}')
            item['status'] = 'error'
            rdb.update(item)