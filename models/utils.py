import json

def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)
    