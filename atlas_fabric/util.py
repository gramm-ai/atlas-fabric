import json, os, random, hashlib

def save_json(path:str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def load_json(path:str):
    with open(path) as f:
        return json.load(f)

def deterministic_rand(seed:int):
    rnd = random.Random(seed)
    return rnd

def sha(s:str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]

def now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")
