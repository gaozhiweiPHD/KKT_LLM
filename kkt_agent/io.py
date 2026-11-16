import json
from pathlib import Path
from .qp_generator import QP

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def load_qps(path,limit=None):
    out=[]
    for row in read_jsonl(path):
        out.append(QP.from_dict(row))
        if limit is not None and len(out)>=limit: break
    return out

def dump_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
