import sys
from functools import cmp_to_key

import pygame
import random
import os.path
import time
import copy
import numpy as np
import random

from pygame.locals import *
from logging import getLogger
from collections import defaultdict
from threading import Thread
from time import sleep
from datetime import datetime

import sources.chess.static_env as senv
from sources.chess.chessboard import Chessboard
from sources.chess.chessman import *
from sources.AlphaZero.ModelManager import ModelManager
from sources.AlphaZero.AI_Player import AI_Player, VisitState
from sources.config import Config
from sources.chess.env import ChessEnv
from sources.chess.lookup_tables import Winner, ActionLabelsRed, flip_move
from sources.utils.modelReaderWriter import load_best_model_weight
from sources.utils.tensorflow_utils import set_session_config
from tkinter import messagebox

from sources.BookHandler.BookHandler import BookHandler
from sources.BookHandler.BookHandler import Book
from sources.BookHandler.BookHandler import BookData
from sources.utils.uci_engine import Engine_Manager

logger = getLogger(__name__)
main_dir = os.path.split(os.path.abspath(__file__))[0]


def start(config: Config, human_move_first=True):
    global PIECE_STYLE
    play = PVE(config)
    play.start(human_move_first)


def Bookcmp(bd1: BookData, bd2: BookData):
    if bd1.score != bd2.score:
        return 1 if bd1.score > bd2.score else -1
    elif bd1.win_rate != bd2.win_rate:
        return 1 if bd1.win_rate > bd2.win_rate else -1
    elif bd1.win_num != bd2.win_num:
        return 1 if bd1.win_num > bd2.win_num else -1
    else:
        return 0


class PVE:
    def __init__(self, config: Config):
        self.config = config
        self.env = ChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.winstyle = 0
        self.chessmans = None
        self.human_move_first = True
        self.screen_width = 521
        self.screen_height = 720
        self.height = 577
        self.width = 521
        self.chessman_w = 57
        self.chessman_h = 57
        self.disp_record_num = 8
        self.rec_labels = [None] * self.disp_record_num
        self.nn_value = 0
        self.mcts_moves = {}
        self.history = []
        self.chessman_history = []
        self.record_history = []
        self.turn_history = []
        self.chessman_hash_history = []
        self.bookhandler = Book(self.config.resource.book_path)
        self.BsetMove = []
        self.book_msg = None
        self.info_msg = None
        self.moves_history = []
        # 初始化红黑双方的倒计时（45 分钟 = 2700 秒）
        self.red_time_left = 2700
        self.black_time_left = 2700
        self.current_timer = None  # 当前正在计时的一方
        self.last_move_time = None  # 上一次移动的时间

    def TranslateMove(self, move):
        return chr(int(move[0]) + ord('a')) + str(move[1]) + chr(int(move[2]) + ord('a')) + str(move[3])

    def hittest(self, mouse_x, mouse_y, rect):
        '''检测鼠标点击事件'''
        # 如果鼠标点击在矩形区域内
        # 则返回True，否则返回False
        if (mouse_x > rect[0] and mouse_x < rect[0] + rect[2] and
                mouse_y > rect[1] and mouse_y < rect[1] + rect[3]):
            return True
        else:
            return False

    def load_model(self):
        self.model = ModelManager(self.config)
        if not load_best_model_weight(self.model):
            self.model.build()

    def init_screen(self):
        bestdepth = pygame.display.mode_ok([self.screen_width, self.screen_height], self.winstyle, 32)
        screen = pygame.display.set_mode([self.screen_width, self.screen_height], self.winstyle, bestdepth)
        pygame.display.set_caption("SYNU AI Chess")
        # create the background, tile the bgd image
        bgdtile = load_image('Board.GIF')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.height])
        board_background.blit(bgdtile, (0, 0))
        # widget_background = pygame.Surface([self.screen_width - self.width, self.height])
        widget_background = pygame.Surface([self.screen_width, self.screen_height - self.height])
        # white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        white_rect = Rect(0, 0, self.screen_width, self.screen_height - self.height)
        widget_background.fill((255, 255, 255), white_rect)

        # create text label
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, 16)
        font_color = (0, 0, 0)
        font_background = (255, 255, 255)
        t = font.render("Record", True, font_color, font_background)
        t_rect = t.get_rect()
        t_rect.x = 10
        t_rect.y = 10
        widget_background.blit(t, t_rect)

        # 显示红方倒计时
        red_time_text = font.render(f"Red Time: {self.format_time(self.red_time_left)}", True, font_color, font_background)
        red_time_rect = red_time_text.get_rect()
        red_time_rect.x = 10
        red_time_rect.y = 30
        widget_background.blit(red_time_text, red_time_rect)

        # 显示黑方倒计时
        black_time_text = font.render(f"Black Time: {self.format_time(self.black_time_left)}", True, font_color, font_background)
        black_time_rect = black_time_text.get_rect()
        black_time_rect.x = 10
        black_time_rect.y = 50
        widget_background.blit(black_time_text, black_time_rect)

        screen.blit(board_background, (0, 0))
        # screen.blit(widget_background, (self.width, 0))
        screen.blit(widget_background, (0, self.height))

        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w, self.chessman_h)
        return screen, board_background, widget_background

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = AI_Player(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                            enable_resign=True, debugging=True)
        self.human_move_first = human_first

        pygame.init()
        screen, board_background, widget_background = self.init_screen()
        framerate = pygame.time.Clock()

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        current_chessman = None
        if human_first:
            self.env.board.calc_chessmans_moving_list()
            self.current_timer = 'red'  # 红方先开始计时
            self.last_move_time = time.time()

        ai_worker = Thread(target=self.ai_move, name="ai_worker")
        ai_worker.daemon = True
        ai_worker.start()
        sleep(1)
        self.chessman_history.append(copy.deepcopy(self.env.board.chessmans))
        self.chessman_hash_history.append((copy.deepcopy(self.env.board.chessmans_hash)))
        self.record_history.append(copy.deepcopy(self.env.board.record))
        self.turn_history.append(copy.deepcopy(self.env.board.turns))

        def undo_move():
            if (len(self.chessman_history) > (2 if not self.human_move_first else 1)):
                for i in range(4):
                    self.history.pop()
                for i in range(2):
                    self.moves_history.pop()
                self.chessman_history.pop()
                self.record_history.pop()
                self.turn_history.pop()
                self.chessman_hash_history.pop()
                # self.gui_history.pop()
                # self.env.board.__chessmans_hash = copy.deepcopy(self.chessman_history[-1])
                self.env.board.Set_chessmans(copy.deepcopy(self.chessman_history[-1]))
                self.env.board.Set_chessmans_hash(copy.deepcopy(self.chessman_hash_history[-1]))
                self.env.board.turns = self.turn_history[-1]
                self.env.board.record = self.record_history[-1]
                self.chessmans.empty()
                for chess in self.env.board.chessmans_hash.values():
                    if chess.is_red:
                        if isinstance(chess, Rook):
                            images = load_images("RR.GIF", "RRS.GIF")
                        elif isinstance(chess, Cannon):
                            images = load_images("RC.GIF", "RCS.GIF")
                        elif isinstance(chess, Knight):
                            images = load_images("RN.GIF", "RNS.GIF")
                        elif isinstance(chess, King):
                            images = load_images("RK.GIF", "RKS.GIF")
                        elif isinstance(chess, Elephant):
                            images = load_images("RB.GIF", "RBS.GIF")
                        elif isinstance(chess, Mandarin):
                            images = load_images("RA.GIF", "RAS.GIF")
                        else:
                            images = load_images("RP.GIF", "RPS.GIF")
                    else:
                        if isinstance(chess, Rook):
                            images = load_images("BR.GIF", "BRS.GIF")
                        elif isinstance(chess, Cannon):
                            images = load_images("BC.GIF", "BCS.GIF")
                        elif isinstance(chess, Knight):
                            images = load_images("BN.GIF", "BNS.GIF")
                        elif isinstance(chess, King):
                            images = load_images("BK.GIF", "BKS.GIF")
                        elif isinstance(chess, Elephant):
                            images = load_images("BB.GIF", "BBS.GIF")
                        elif isinstance(chess, Mandarin):
                            images = load_images("BA.GIF", "BAS.GIF")
                        else:
                            images = load_images("BP.GIF", "BPS.GIF")
                    chessman_sprite = Chessman_Sprite(images, chess, self.chessman_w, self.chessman_h)
                    self.chessmans.add(chessman_sprite)
                # self.chessmans = self.gui_history[-1]

                self.draw_widget(screen, widget_background)
                framerate.tick(20)
                # clear/erase the last drawn sprites
                self.chessmans.clear(screen, board_background)

                # update all the sprites
                self.chessmans.update()
                self.chessmans.draw(screen)
                pygame.display.update()
            else:
                print("不存在可以悔棋的历史局面!")

        while not self.env.board.is_end():
            if self.current_timer:
                elapsed_time = time.time() - self.last_move_time
                if self.current_timer == 'red':
                    self.red_time_left -= elapsed_time
                    if self.red_time_left <= 0:
                        self.env.board.winner = Winner.black
                        break
                else:
                    self.black_time_left -= elapsed_time
                    if self.black_time_left <= 0:
                        self.env.board.winner = Winner.red
                        break
                self.last_move_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.board.print_record()
                    self.ai.close(wait=False)
                    game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(self.config.resource.play_record_dir,
                                        self.config.resource.play_record_filename_tmpl % game_id)
                    self.env.board.save_record(path)
                    sleep(10)
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    pass
                elif event.type == MOUSEBUTTONDOWN:
                    if human_first == self.env.red_to_move:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        # print(mouse_x, mouse_y)
                        if self.hittest(mouse_x, mouse_y, (150, 600, 120, 50)):
                            print("悔棋!\n")
                            undo_move()
                        elif self.hittest(mouse_x, mouse_y, (150, 700, 120, 50)):
                            print("打印棋谱!")
                            self.env.board.print_record()
                            game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                            path = os.path.join(self.config.resource.play_record_dir,
                                                self.config.resource.play_record_filename_tmpl % game_id)
                            self.env.board.save_record(path)
                        else:
                            pressed_array = pygame.mouse.get_pressed()
                            for index in range(len(pressed_array)):
                                if index == 0 and pressed_array[index]:
                                    mouse_x, mouse_y = pygame.mouse.get_pos()
                                    col_num, row_num = translate_hit_area(mouse_x, mouse_y, self.chessman_w,
                                                                          self.chessman_h)
                                    chessman_sprite = select_sprite_from_group(
                                        self.chessmans, col_num, row_num)
                                    if current_chessman is None and chessman_sprite != None:
                                        if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                            current_chessman = chessman_sprite
                                            chessman_sprite.is_selected = True
                                    elif current_chessman != None and chessman_sprite != None:
                                        if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                            current_chessman.is_selected = False
                                            current_chessman = chessman_sprite
                                            chessman_sprite.is_selected = True
                                        else:
                                            move = str(current_chessman.chessman.col_num) + str(
                                                current_chessman.chessman.row_num) + \
                                                   str(col_num) + str(row_num)
                                            success = current_chessman.move(col_num, row_num, self.chessman_w,
                                                                            self.chessman_h)
                                            self.history.append(move)
                                            self.moves_history.append(self.TranslateMove(move))
                                            # print(self.moves_history)
                                            if success:
                                                self.chessmans.remove(chessman_sprite)
                                                chessman_sprite.kill()
                                                current_chessman.is_selected = False
                                                current_chessman = None
                                                self.history.append(self.env.get_state())
                                                # 切换到黑方计时
                                                self.current_timer = 'black'
                                                self.last_move_time = time.time()
                                    elif current_chessman != None and chessman_sprite is None:
                                        move = str(current_chessman.chessman.col_num) + str(
                                            current_chessman.chessman.row_num) + \
                                               str(col_num) + str(row_num)
                                        success = current_chessman.move(col_num, row_num, self.chessman_w,
                                                                        self.chessman_h)
                                        self.history.append(move)
                                        self.moves_history.append(self.TranslateMove(move))
                                        # print(self.moves_history)
                                        if success:
                                            current_chessman.is_selected = False
                                            current_chessman = None
                                            self.history.append(self.env.get_state())
                                            # 切换到黑方计时
                                            self.current_timer = 'black'
                                            self.last_move_time = time.time()

            self.draw_widget(screen, widget_background)
            framerate.tick(20)
            # clear/erase the last drawn sprites
            self.chessmans.clear(screen, board_background)

            # update all the sprites
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()

        self.ai.close(wait=False)
        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.config.resource.play_record_dir,
                            self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(path)
        sleep(10)

    def ai_move(self):
        ai_move_first = not self.human_move_first
        self.history = [self.env.get_state()]
        # self.gui_history.append(copy.deepcopy(self.chessmans))
        no_act = None
        while not self.env.done:
            if ai_move_first == self.env.red_to_move:
                labels = ActionLabelsRed
                labels_n = len(ActionLabelsRed)
                self.ai.search_results = {}
                state = self.env.get_state()
                logger.info(f"state = {state}")
                _, _, _, check = senv.done(state, need_check=True)
                if not check and state in self.history[:-1]:
                    no_act = []
                    free_move = defaultdict(int)
                    for i in range(len(self.history) - 1):
                        if self.history[i] == state:
                            # 如果走了下一步是将军或捉：禁止走那步
                            if senv.will_check_or_catch(state, self.history[i + 1]):
                                no_act.append(self.history[i + 1])
                            # 否则当作闲着处理
                            else:
                                free_move[state] += 1
                                # if free_move[state] >= 2:
                                # 作和棋处理
                                #    self.env.winner = Winner.draw
                                #    self.env.board.winner = Winner.draw
                                #    break
                    if no_act:
                        logger.debug(f"no_act = {no_act}")
                BookResult = []
                if not self.config.resource.Use_Book:
                    self.book_msg = "未启用历史局面缓存..."
                if self.config.resource.Use_Book and (
                        self.config.resource.Out_Book_Step == -1 or self.env.board.turns <= self.config.resource.Out_Book_Step):
                    if self.config.resource.Book_Type == 'Local':
                        BookResult = self.bookhandler.query(BookHandler.fixFen(self.env.get_state()), ai_move_first)
                        BookResult.sort(key=cmp_to_key(Bookcmp), reverse=1)

                    else:
                        time.sleep(3)
                        BookResult = BookHandler.get_cloud_move(BookHandler.fixFen(self.env.get_state()), ai_move_first,
                                                                self.config.resource.Cloud_Url)
                    if len(BookResult) > 0:
                        self.book_msg = '命中历史局面!'
                    else:
                        self.book_msg = '未命中历史局面!'

                # for i in BookResult:
                #     print(str(i))
                action = None
                self.BsetMove = []
                if not self.config.resource.Use_EngineHelp:
                    action, policy = self.ai.action(state, self.env.num_halfmoves, no_act)
                    if not self.env.red_to_move:
                        action = flip_move(action)
                if self.config.resource.Use_Book and self.config.resource.Book_Type == 'Local' and len(BookResult) > 0:
                    for i in range(min(3, len(BookResult))):
                        self.BsetMove.append(BookResult[i].move)
                    self.info_msg = '从历史局面获取最优走法:'
                    print("使用库1:", str(BookResult[0].move))
                    x0, y0, x1, y1 = BookHandler.Move2Point(BookResult[0].move)
                    action = str(x0) + str(y0) + str(x1) + str(y1)
                    self.history.append(action)
                    self.moves_history.append(self.TranslateMove(action))
                    # print(self.moves_history)
                elif self.config.resource.Use_Book and self.config.resource.Book_Type == 'Cloud' and len(
                        BookResult) > 0:
                    self.info_msg = '从历史局面获取最优走法:'
                    random.shuffle(BookResult)
                    for i in range(min(3, len(BookResult))):
                        self.BsetMove.append(BookResult[i])
                    print("使用库2:", str(BookResult[0]))
                    x0, y0, x1, y1 = BookHandler.Move2Point(BookResult[0])
                    action = str(x0) + str(y0) + str(x1) + str(y1)
                    self.history.append(action)
                    self.moves_history.append(self.TranslateMove(action))
                    # print(self.moves_history)
                elif self.config.resource.Use_EngineHelp:
                    action = Engine_Manager.get_uci_move(self.config.resource.engine_path,
                                                         self.moves_history,
                                                         self.config.resource.EngineSearchThreads,
                                                         ai_move_first,
                                                         self.config.resource.EngineSearchTime + random.randint(5, 10))
                    self.info_msg = "AlphaZero搜索的最优走法:"
                    self.BsetMove.append(action)
                    # print(action)
                    print("使用AI搜索: ", action)
                    x0, y0, x1, y1 = BookHandler.Move2Point(action)
                    action = str(x0) + str(y0) + str(x1) + str(y1)
                    self.history.append(action)
                    self.moves_history.append(self.TranslateMove(action))
                    # print(self.moves_history)
                else:
                    self.BsetMove.append(action)
                    self.info_msg = "MCTS搜索的最优走法:"
                    print("使用MCTS搜索: ", action)
                    self.history.append(action)
                    self.moves_history.append(self.TranslateMove(action))
                    # print(self.moves_history)
                if action is None:
                    return
                if not self.config.resource.Use_EngineHelp:
                    key = self.env.get_state()
                    p, v = self.ai.debug[key]
                    logger.info(f"check = {check}, NN value = {v:.3f}")
                    self.nn_value = v
                    logger.info("MCTS results:")
                    self.mcts_moves = {}
                    for move, action_state in self.ai.search_results.items():
                        move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]),
                                                                    int(move[3]))
                        logger.info(
                            f"move: {move_cn}-{move}, visit count: {action_state[0]}, Q_value: {action_state[1]:.3f}, Prior: {action_state[2]:.3f}")
                        self.mcts_moves[move_cn] = action_state
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
                if sprite_dest:
                    self.chessmans.remove(sprite_dest)
                    sprite_dest.kill()
                chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
                self.history.append(self.env.get_state())
                self.chessman_history.append(copy.deepcopy(self.env.board.chessmans))
                self.record_history.append(copy.deepcopy(self.env.board.record))
                self.turn_history.append(copy.deepcopy(self.env.board.turns))
                self.chessman_hash_history.append((copy.deepcopy(self.env.board.chessmans_hash)))
                # 切换到红方计时
                self.current_timer = 'red'
                self.last_move_time = time.time()

    def draw_widget(self, screen, widget_background):
        # white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        white_rect = Rect(0, 0, self.screen_width, self.screen_height - self.height)
        widget_background.fill((255, 255, 255), white_rect)
        # pygame.draw.line(widget_background, (255, 0, 0), (10, 285), (self.screen_width - self.width - 10, 285))
        # screen.blit(widget_background, (self.width, 0))
        screen.blit(widget_background, (0, self.height))

        self.draw_records(screen, widget_background)
        if not self.config.resource.Use_EngineHelp:
            self.draw_evaluation(screen, widget_background)
        else:
            self.draw_label(screen, widget_background, '--------------INFO-------------', 0, 14, self.screen_width - 240)
            self.draw_label(screen, widget_background, self.book_msg, 18, 14, self.screen_width - 240)
            self.draw_label(screen, widget_background, self.info_msg, 36, 14, self.screen_width - 240)
            for i in range(min(len(self.BsetMove), 3)):
                self.draw_label(screen, widget_background, str(i) + ' :    ' + self.BsetMove[i], 55 + i * 18, 14, self.screen_width - 240)
            if len(self.BsetMove) > 0:
                self.draw_label(screen, widget_background, '执行: ' + self.BsetMove[0], 55 + 3 * 18, 14, self.screen_width - 240)
        self.draw_undo_button(screen, widget_background)
        nowDate, nowTime = time.strftime('%Y-%m-%d %H:%M:%S').split(' ')
        self.draw_label(screen, widget_background, "当前时间:", 70, 15, 140)
        self.draw_label(screen, widget_background, nowDate, 70, 15, 140)
        self.draw_label(screen, widget_background, nowTime, 90, 15, 140)
        self.draw_label(screen, widget_background, "打印棋谱", 120, 15, 150)

        # 显示红方倒计时
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, 16)
        font_color = (0, 0, 0)
        font_background = (255, 255, 255)
        red_time_text = font.render(f"Red Time: {self.format_time(self.red_time_left)}", True, font_color, font_background)
        red_time_rect = red_time_text.get_rect()
        red_time_rect.x = 10
        red_time_rect.y = 30
        widget_background.blit(red_time_text, red_time_rect)

        # 显示黑方倒计时
        black_time_text = font.render(f"Black Time: {self.format_time(self.black_time_left)}", True, font_color, font_background)
        black_time_rect = black_time_text.get_rect()
        black_time_rect.x = 10
        black_time_rect.y = 50
        widget_background.blit(black_time_text, black_time_rect)

        screen.blit(widget_background, (0, self.height))

    def draw_undo_button(self, screen, widget_background):
        self.draw_label(screen, widget_background, "悔棋", 15, 20, 150)

    def draw_records(self, screen, widget_background):
        text = 'Record'
        self.draw_label(screen, widget_background, text, 0, 16, 10)
        records = self.env.board.record.split('\n')
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, 12)
        i = 0
        for record in records[-self.disp_record_num:]:
            self.rec_labels[i] = font.render(record, True, (0, 0, 0), (255, 255, 255))
            t_rect = self.rec_labels[i].get_rect()
            # t_rect.centerx = (self.screen_width - self.width) / 2
            t_rect.y = 25 + i * 15
            t_rect.x = 10
            t_rect.width = self.screen_width - self.width
            widget_background.blit(self.rec_labels[i], t_rect)
            i += 1
        # screen.blit(widget_background, (self.width, 0))
        screen.blit(widget_background, (0, self.height))

    def draw_evaluation(self, screen, widget_background):
        # title_label = 'Info:'
        # self.draw_label(screen, widget_background, title_label, 10, 16, 50)
        info_label = f'MCTS simulation：{self.config.play.simulation_num_per_move}'
        self.draw_label(screen, widget_background, info_label, 0, 14, self.screen_width - 240)
        eval_label = f"Now Value: {self.nn_value:.3f}"
        self.draw_label(screen, widget_background, eval_label, 15, 14, self.screen_width - 240)
        label = f"Result:"
        self.draw_label(screen, widget_background, label, 30, 14, self.screen_width - 240)
        label = f"Method | Count | ActionValue | Probability"
        self.draw_label(screen, widget_background, label, 45, 12, self.screen_width - 240)
        i = 0
        tmp = copy.deepcopy(self.mcts_moves)
        for mov, action_state in tmp.items():
            label = f"{mov}"
            self.draw_label(screen, widget_background, label, 60 + i * 20, 12, self.screen_width - 235)
            label = f"{action_state[0]}"
            self.draw_label(screen, widget_background, label, 60 + i * 20, 12, self.screen_width - 180)
            label = f"{action_state[1]:.2f}"
            self.draw_label(screen, widget_background, label, 60 + i * 20, 12, self.screen_width - 120)
            label = f"{action_state[2]:.3f}"
            self.draw_label(screen, widget_background, label, 60 + i * 20, 12, self.screen_width - 50)
            i += 1

    def draw_label(self, screen, widget_background, text, y, font_size, x=None):
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, font_size)
        label = font.render(text, True, (0, 0, 0), (255, 255, 255))
        t_rect = label.get_rect()
        t_rect.y = y
        if x != None:
            t_rect.x = x
        else:
            t_rect.centerx = (self.screen_width - self.width) / 2
        widget_background.blit(label, t_rect)
        # screen.blit(widget_background, (self.width, 0))
        screen.blit(widget_background, (0, self.height))

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


class Chessman_Sprite(pygame.sprite.Sprite):
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman, w=80, h=80):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = [pygame.transform.scale(image, (w, h)) for image in images]
        self.image = self.images[0]
        self.rect = Rect(chessman.col_num * w, (9 - chessman.row_num) * h, w, h)

    def move(self, col_num, row_num, w=80, h=80):
        # print self.chessman.name, col_num, row_num
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect = Rect(old_col_num * w, (9 - old_row_num) * h, w, h)
            self.rect.move_ip((col_num - old_col_num)
                              * w, (old_row_num - row_num) * h)
            # self.rect = self.rect.clamp(SCREENRECT)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            self.image = self.images[1]
        else:
            self.image = self.images[0]


def load_image(file, sub_dir=None):
    '''loads an image, prepares it for play'''
    if sub_dir:
        file = os.path.join(main_dir, 'images', sub_dir, file)
    else:
        file = os.path.join(main_dir, 'images', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s' %
                         (file, pygame.get_error()))
    return surface.convert()


def load_images(*files):
    imgs = []
    for file in files:
        imgs.append(load_image(file, 'Piece'))
    return imgs


def creat_sprite_group(sprite_group, chessmans_hash, w, h):
    for chess in chessmans_hash.values():
        if chess.is_red:
            if isinstance(chess, Rook):
                images = load_images("RR.GIF", "RRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("RC.GIF", "RCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("RN.GIF", "RNS.GIF")
            elif isinstance(chess, King):
                images = load_images("RK.GIF", "RKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("RB.GIF", "RBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.GIF", "RAS.GIF")
            else:
                images = load_images("RP.GIF", "RPS.GIF")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.GIF", "BRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("BC.GIF", "BCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("BN.GIF", "BNS.GIF")
            elif isinstance(chess, King):
                images = load_images("BK.GIF", "BKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("BB.GIF", "BBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.GIF", "BAS.GIF")
            else:
                images = load_images("BP.GIF", "BPS.GIF")
        chessman_sprite = Chessman_Sprite(images, chess, w, h)
        sprite_group.add(chessman_sprite)


def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None


def translate_hit_area(screen_x, screen_y, w=80, h=80):
    return screen_x // w, 9 - screen_y // h
