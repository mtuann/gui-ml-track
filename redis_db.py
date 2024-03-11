import os
import json
import uuid
import redis
from datetime import datetime

class RedisDB:

    qreqs = 'queue:requests'

    def __init__(self, url, db):
        self.db = redis.from_url(url=url, db=db)

    def next_request(self):
        data = self.db.blpop(self.qreqs, 1)
        if not data:
            return None
        data = json.loads(data[1])
        return data

    def add(self, hyper_params):
        data = self._make_data(hyper_params)
        
        self.db.rpush(self.qreqs, json.dumps(data))
        
        ret = self.update(data)
        return ret

    def update(self, new_data):
        iid = new_data.get('iid', None)
        if iid is None: return None
        try:
            if self.db.exists(iid):
                self.db.delete(iid)
            self.db.set(iid, json.dumps(new_data))
            return iid
        except:
            return None

    def get_iid_status(self, iid):
        out = self.db.get(iid)
        if out is None: return {}
        out = out.decode('utf-8')
        out = json.loads(out)
        return out
    
    def get_all_iid_status(self):
        keys = self.db.keys()
        # print(keys)
        out = {
            "iid": [],
            "status": [],
            "hyper_params": [],
            "time_added" : [],
        }
        for key in keys:
            key = key.decode('utf-8')
            if key.startswith('queue:'):
                continue
            
            data_iid = self.get_iid_status(key)
            if "status" in data_iid and "in_update" in data_iid:
                out["iid"].append(key)
                out["status"].append(data_iid['status'])
                out["hyper_params"].append(data_iid['hyper_params'])
                out["time_added"].append(data_iid.get('time_added', ''))
            
        return out

    def clear_queue(self):
        n = self.db.llen(self.qreqs)
        for _ in range(n):
            self.db.blpop(self.qreqs)

    def _make_data(self, hyper_params):
        iid = uuid.uuid4().hex
        # status: error|waiting|done
        data = {
            'iid': iid, 
            'hyper_params': hyper_params,
            'status': 'waiting',
            'in_update': [],
            "time_added": f"{datetime.now()}"
        }
        return data