import json
import jsonschema
import os

schema_path = os.path.join(os.path.dirname(__file__), "schema", "cfg.schema.json")

def load_cfg(cfg_path: str) -> json:
    f_cfg = open(os.path.abspath(cfg_path), "r")
    cfg = json.load(f_cfg)
    f_cfg.close()
    return cfg

def validate_cfg(cfg: json, schema: json) -> bool:
    try:
        jsonschema.validate(instance=cfg, schema=schema)
    except:
        return False
    return True

if __name__ == '__main__':
    print("JSON Config")
    print(schema_path)
    default_cfg = os.path.abspath(os.path.join("..", "config", "default_cfg.json"))
    print(default_cfg)
    print("Load Config")
    cfg = load_cfg(default_cfg)
    print("Algoritm type")
    print(cfg['optimizer']['algorithm'])

