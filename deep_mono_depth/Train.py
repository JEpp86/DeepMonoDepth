import os
import argparse
import logging
from deep_mono_depth.core.Trainer import Trainer
from deep_mono_depth.util.json_cfg import Config


def main():
    parser = argparse.ArgumentParser(description="Train Deep Mono Depth Model")
    parser.add_argument("--config", "-c", type=str, help="training configuration", required=True)
    parser.add_argument("--logfile", "-l", type=str, help="output log file")
    args = parser.parse_args()
    # TODO logging not great, need to invstigate
    """
    if args.logfile:
        logging.basicConfig(
            filename=args.logfile,
            encoding="utf-8",
            level=logging.INFO,
            format="[%(levelname)s](%(name)s) %(asctime)s - %(message)s",
        )
    """

    cfg = Config(os.path.abspath(args.config))
    cfg.setup_logging()
    logging.info("Deep Mono Depth - Train")
    logging.info("Loading Config: " + args.config)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
