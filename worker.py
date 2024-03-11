import os
import json
import time
import numpy as np

# from config import get_env_config
from redis_db import RedisDB
from train_mnist import training_and_update


# CONFIG = get_env_config()
rdb = RedisDB('redis://localhost:6379', 0)

def run(item):
    
    # hyper_params = {
    #     "optimizer": "adam",
    #     "learning_rate": 0.001,
    #     "num_epochs": 5
    # }
    hyper_params = item['hyper_params']

    meta_data = training_and_update(hyper_params, rdb, item)


if __name__ == '__main__':
    while True:
        item = rdb.next_request()
        if not item:
            print('Request Queue is empty!', end='\r')
            time.sleep(1)
            continue

        try:
            print(f'Processing item: {item["iid"]}')
            run(item)
        except Exception as e:
            item['status'] = 'error'
            rdb.update(item)