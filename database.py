import yaml
import redis

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['database']

# Connect to Redis server
try:
    r = redis.StrictRedis(host=config['host'], port=config['port'], db=config['db'])
    r.ping()  # Check if the connection is successful
    print("Connected to Redis server successfully!")
except redis.ConnectionError:
    print("Failed to connect to Redis server")
