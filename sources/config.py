import os
import getpass
import config as cfg

def _project_dir():
    d = os.path.dirname
    return d(d(os.path.abspath(__file__)))


def _data_dir():
    return os.path.join(_project_dir(), "data")

class Config:
    def __init__(self):
        self.trainsetting = trainsetting()
        self.resource = ResourceConfig()

        self.model = cfg.ModelConfig()
        self.play = cfg.PlayConfig()
        self.play_data = cfg.GenerateDataConfig()
        self.trainer = cfg.TrainerConfig()

class trainsetting:
    new = False
    light = True
    device_list = '0'
    random = 'none'
    log_move = False
    use_multiple_gpus = False
    gpu_num = 1
    evaluate = False
    has_history = False

class GenerateDataConfig:
    def __init__(self):
        self.sl_nb_game_in_file = 250
        self.nb_game_in_file = 5
        self.max_file_num = 3000
        self.nb_game_save_record = 1

class PVEConfig:
    def __init__(self):
        self.simulation_num_per_move = 3000
        self.c_puct = 1
        self.search_threads = 32
        self.noise_eps = 0
        self.tau_decay_rate = 0
        self.dirichlet_alpha = 0.2

    def update_play_config(self, pc):
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.tau_decay_rate = self.tau_decay_rate
        pc.search_threads = self.search_threads
        pc.dirichlet_alpha = self.dirichlet_alpha

class PlayConfig:
    def __init__(self):
        self.max_processes = 32
        self.search_threads = 64
        self.vram_frac = 1.0
        self.simulation_num_per_move = 3200
        self.thinking_loop = 1
        self.logging_thinking = True
        self.c_puct = 1.5
        self.noise_eps = 0.15
        self.dirichlet_alpha = 0.2
        self.tau_decay_rate = 0.9
        self.virtual_loss = 3
        self.resign_threshold = -0.98
        self.min_resign_turn = 40
        self.enable_resign_rate = 0.5
        self.max_game_length = 200
        self.share_mtcs_info_in_self_play = False
        self.reset_mtcs_info_per_game = 5


class TrainerConfig:
    def __init__(self):
        self.min_games_to_begin_learn = 100
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 4
        self.vram_frac = 1.0
        self.batch_size = 512
        self.epoch_to_checkpoint = 3
        self.dataset_size = 100000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        self.momentum = 0.9
        self.loss_weights = [1.0, 1.0]
        self.lr_schedules = [
            (0, 0.01),
            (150000, 0.003),
            (400000, 0.0001),
        ]
        self.sl_game_step = 2000
        self.load_step = 6

class ModelConfig:
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_first_filter_size = 5
        self.cnn_filter_size = 3
        self.res_layer_num = 7
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.distributed = False
        self.input_depth = 14

class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())

        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")
        self.eleeye_path = os.path.join(self.model_dir, 'eleeye.exe')
        self.engine_path = os.path.join(self.project_dir, 'data', 'Engine', 'pikafish')

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_config_path = os.path.join(self.next_generation_model_dir, "next_generation_config.json")
        self.next_generation_weight_path = os.path.join(self.next_generation_model_dir, "next_generation_weight.h5")
        self.rival_model_config_path = os.path.join(self.model_dir, "rival_config.json")
        self.rival_model_weight_path = os.path.join(self.model_dir, "rival_weight.h5")

        self.play_data_dir = os.path.join(self.data_dir, "train_data")
        self.play_data_filename_tmpl = "play_%s.json"
        self.self_play_game_idx_file = os.path.join(self.data_dir, "play_data_idx")
        self.play_record_filename_tmpl = "record_%s.pgn"
        self.play_record_dir = os.path.join(self.data_dir, "play_record")

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.train_log_path = os.path.join(self.log_dir, "train.log")
        self.play_log_path = os.path.join(self.log_dir, "play.log")

        self.font_path = os.path.join(self.project_dir, 'sources', 'game', 'font', 'font.ttc')

        self.Use_Book = True
        self.Out_Book_Step = -1

        self.Use_EngineHelp = True # 使用引擎辅助计算
        self.EngineSearchThreads = 32 # 引擎线程数
        self.EngineSearchTime = 5 # 引擎时间
        self.book_path = os.path.join(self.project_dir, 'data', 'Books', 'BOOK1.obk') # 本地库位置

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.play_record_dir, self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
