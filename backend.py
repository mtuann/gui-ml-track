import yaml
from flask import Flask

app = Flask(__name__)

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['backend']

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host=config['host'], port=config['port'], debug=True)
