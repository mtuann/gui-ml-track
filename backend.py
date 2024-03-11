import yaml
from flask import Flask
from flask import request
from redis_db import RedisDB

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['database']

app = Flask(__name__)
rdb = RedisDB(f'redis://{config["host"]}:{config["port"]}', config['db'])


@app.route('/')
def hello():
    return 'Hello, World!'

# Add more routes here
# training round for machine learning model

# get data from request of user
@app.route('/train', methods=['POST'])
def train():
    print(f"Parameters from user request: {request.json}")
    # train_hyper_params(request.json)
    # hyper_params = {
    #     "optimizer": "adam",
    #     "learning_rate": 0.001,
    #     "num_epochs": 5
    # }
    params = request.json
    ret = rdb.add(params)
    if ret is not None:
        return {'success': True, 'iid': ret}
    else:
        return {'success': False, 'iid': ""}
    
# get status from iid
@app.route('/status/<iid>', methods=['GET'])
def get_status(iid: str):
    out = rdb.get_iid_status(iid)
    return out
    # sample: curl -X GET http://localhost:5000/status/fd19d1a65f9e48d8b91bfa99061aea42    

# get list of experiments
@app.route('/experiments', methods=['GET'])
def get_experiments():
    out = rdb.get_all_iid_status()
    return out
    # sample: curl -X GET http://localhost:5000/experiments

if __name__ == '__main__':
    app.run(host=config['host'], port=config['port'], debug=True)
