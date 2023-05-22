import os
import argparse
import util.json_cfg as config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Train Deep Mono Depth Model")
    parser.add_argument('--config','-c', type=str, help="training configuration", required=True)
    args = parser.parse_args()
    
    cfg = config.load_cfg(os.path.abspath(args.config))
    print(cfg)