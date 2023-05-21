import json
import jsonschema
import os

schema_path = os.path.join(os.path.dirname(__file__), "schema", "cfg.schema.json")

def load_cfg(cfg_path: str) -> json:
    pass

def validate_cfg(cfg: json, schema: json) -> bool:
    try:
        jsonschema.validate(instance=cfg, schema=schema)
    except:
        return False
    return True

def parse_cfg(cfg: json) -> dict[str, any]:
    pass

if __name__ == '__main__':
    import argparse
    print("JSON Config Tests")
    print(schema_path)