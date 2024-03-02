import yaml
from flask import Flask
from flask import request

app = Flask(__name__)

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['backend']

@app.route('/')
def hello():
    return 'Hello, World!'

# Add more routes here
# training round for machine learning model
# @app.route('/train')
# def train():
#     print(f"Parameters from user request: {request.args}")
#     return 'Training...'

# get data from request of user
@app.route('/train', methods=['POST'])
def train():
    print(f"Parameters from user request: {request.json}")
    return {'status': 'Training...'}

if __name__ == '__main__':
    app.run(host=config['host'], port=config['port'], debug=True)
