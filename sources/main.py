import logging
import os
import sys
import argparse
import multiprocessing as mp

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from logging import getLogger

from sources.utils.logger import setup_logger
from sources.config import Config, PVEConfig


logger = getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

CMD_LIST = ['generate_data', 'train', 'play', 'play_to_self']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=CMD_LIST)
    parser.add_argument("--ai-move-first", action="store_true")
    parser.add_argument("--ucci", action="store_true")
    return parser

def setup(config: Config, args):
    config.resource.create_directories()
    if args.cmd == 'generate_data':
        setup_logger(config.resource.main_log_path)
    elif args.cmd == 'train':
        setup_logger(config.resource.train_log_path)
    elif args.cmd == 'play' or args.cmd == 'play_to_self':
        setup_logger(config.resource.play_log_path)

def start():
    parser = create_parser()
    args = parser.parse_args()

    config = Config()
    setup(config, args)

    if args.cmd == 'generate_data':
        if args.ucci:
            import sources.worker.TrainWithUCCI as self_play
        else:
            if mp.get_start_method() == 'spawn':
                import sources.worker.TrainDataGenerater_win as self_play
            else:
                from sources.worker import TrainDataGenerater
        return self_play.start(config)
    elif args.cmd == 'train':
        from sources.worker import Train
        return Train.start(config)
    elif args.cmd == 'play':
        from sources.game import play
        config.trainsetting.light = False
        PlayConfig = PVEConfig()
        PlayConfig.update_play_config(config.play)
        logger.info(f"AI move first : {args.ai_move_first}")
        play.start(config, not args.ai_move_first)
    elif args.cmd == 'play_to_self':
        from sources.game import PlayToSelf
        PlayConfig = PVEConfig()
        PlayConfig.update_play_config(config.play)
        PlayToSelf.start(config, args.ucci, args.ai_move_first)

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    start()