from multiprocessing import connection, Pipe
from threading import Thread

import os
import numpy as np
import shutil

from sources.config_enhanced import EnhancedConfig as Config
from sources.utils.modelReaderWriter import load_best_model_weight, need_to_reload_best_model_weight
from time import time
from logging import getLogger

logger = getLogger(__name__)

class Predictor:

    def __init__(self, config: Config, _model):
        self.model = _model
        self.pipes = []
        self.config = config
        self.need_reload = True
        self.done = False

    def start(self, need_reload=True):
        self.need_reload = need_reload
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self, need_reload=True):
        me, you = Pipe()
        self.pipes.append(me)
        self.need_reload = need_reload
        return you

    def predict_batch_worker(self):
        last_model_check_time = time()
        while not self.done:
            if last_model_check_time + 600 < time() and self.need_reload:
                self.try_reload_model()
                last_model_check_time = time()
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes, data_len = [], [], []
            for pipe in ready:
                while pipe.poll():
                    try:
                        tmp = pipe.recv()
                    except EOFError as e:
                        logger.error(f"EOF error: {e}")
                        pipe.close()
                    else:
                        data.extend(tmp)
                        data_len.append(len(tmp))
                        result_pipes.append(pipe)
            if not data:
                continue
            data = np.asarray(data, dtype=np.float32)
            # In TensorFlow 2.x, we don't need to use graph context
            policy_ary, value_ary = self.model.model.predict_on_batch(data)
            buf = []
            k, i = 0, 0
            for p, v in zip(policy_ary, value_ary):
                buf.append((p, float(v)))
                k += 1
                if k >= data_len[i]:
                    result_pipes[i].send(buf)
                    buf = []
                    k = 0
                    i += 1

    def try_reload_model(self, config_file=None):
        if config_file:
            config_path = os.path.join(self.config.resource.model_dir, config_file)
            shutil.copy(config_path, self.config.resource.model_best_config_path)
        try:
            if self.need_reload and need_to_reload_best_model_weight(self.model):
                # In TensorFlow 2.x, we don't need to use graph context
                load_best_model_weight(self.model)
        except Exception as e:
            logger.error(e)

    def close(self):
        self.done = True
