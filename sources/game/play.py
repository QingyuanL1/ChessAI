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

logger = getLogger(__name__)
from sources.chess.chessboard import Chessboard
from sources.chess.chessman import *
from sources.AlphaZero.ModelManager import ModelManager
from sources.AlphaZero.Enhanced_AI_Player import Enhanced_AI_Player as AI_Player, EnhancedVisitState as VisitState
from sources.config_enhanced import EnhancedConfig as Config
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


def start(config: Config, human_move_first=None):
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
        self.screen_width = 700
        self.screen_height = 850
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

        # 美化主题色彩配置 - 豪华深色主题
        self.colors = {
            'bg_primary': (18, 25, 38),  # 更深的主背景
            'bg_secondary': (28, 37, 54),  # 渐变次级背景
            'bg_tertiary': (40, 51, 73),  # 第三级背景
            'bg_panel': (52, 67, 97),  # 面板背景（优雅蓝）
            'bg_panel_hover': (62, 80, 115),  # 面板悬停色
            'bg_elevated': (72, 89, 128),  # 提升元素背景
            'accent': (255, 193, 7),  # 金黄色强调（更优雅）
            'accent_hover': (255, 213, 79),  # 强调色悬停
            'accent_secondary': (79, 172, 254),  # 亮蓝色辅助强调
            'accent_tertiary': (139, 92, 246),  # 紫色强调
            'text_primary': (248, 250, 252),  # 纯白主要文字
            'text_secondary': (186, 199, 216),  # 次要文字
            'text_muted': (125, 145, 174),  # 静音文字
            'text_accent': (255, 193, 7),  # 强调文字
            'success': (34, 197, 94),  # 鲜绿成功色
            'warning': (245, 158, 11),  # 橙色警告
            'error': (239, 68, 68),  # 红色错误
            'info': (59, 130, 246),  # 蓝色信息
            'red_player': (220, 38, 127),  # 红方玩家（玫红）
            'black_player': (31, 41, 55),  # 黑方玩家（深蓝灰）
            'border': (71, 85, 105),  # 边框色
            'border_light': (148, 163, 184),  # 浅边框色
            'border_accent': (255, 193, 7),  # 强调边框
            'shadow': (0, 0, 0, 120),  # 深阴影
            'shadow_light': (0, 0, 0, 60),  # 浅阴影
            'shadow_colored': (0, 0, 0, 40),  # 彩色阴影
            'glass': (255, 255, 255, 25),  # 玻璃效果
            'glass_dark': (255, 255, 255, 10),  # 深玻璃效果
            'highlight': (255, 255, 255, 15),  # 高光效果
        }

        # 游戏统计数据
        self.game_stats = {
            'total_games': 0,
            'player_wins': 0,
            'ai_wins': 0,
            'draws': 0,
            'total_moves': 0,
            'avg_move_time': 0,
            'current_game_start': None,
            'current_game_end': None,
            'thinking_time': 0,
            'positions_evaluated': 0
        }

        # 操作状态反馈
        self.operation_feedback = {
            'message': '',
            'timestamp': 0,
            'type': 'info'  # 'info', 'success', 'warning', 'error'
        }

    def set_operation_feedback(self, message, feedback_type='info'):
        """设置操作反馈信息"""
        self.operation_feedback = {
            'message': message,
            'timestamp': time.time(),
            'type': feedback_type
        }

    # def check_threefold_repetition(self):
    #     """检查三次重复局面，符合中国象棋规则 - 已禁用"""
    #     if len(self.history) < 6:  # 至少需要6步才可能出现三次重复（每方各3步）
    #         return False
    #         
    #     current_state = self.env.get_state()
    #     
    #     # 统计当前局面在历史中出现的次数（包括当前）
    #     state_count = self.history.count(current_state)
    #     
    #     # 如果当前状态还没有加入history，需要+1
    #     if current_state not in self.history:
    #         state_count += 1
    #         
    #     if state_count >= 3:
    #         logger.info(f"🔄 检测到三次重复局面，判定和棋 (出现{state_count}次)")
    #         self.env.winner = Winner.draw
    #         self.env.board.winner = Winner.draw  
    #         self.set_operation_feedback("🔄 三次重复局面，自动判和！", 'info')
    #         print("检测到三次重复局面，自动判和！")
    #         return True
    #         
    #     # 记录调试信息
    #     if state_count >= 2:
    #         logger.debug(f"重复局面检测: 当前局面已出现{state_count}次，总步数={len(self.history)}")
    #         
    #     return False

    def is_human_turn(self):
        """判断当前是否轮到人类玩家"""
        # 如果人类是红方先走，那么红方轮次时是人类回合
        # 如果AI是红方先走，那么黑方轮次时是人类回合
        is_human = False
        if self.human_move_first:
            is_human = self.env.red_to_move  # 人类是红方
        else:
            is_human = not self.env.red_to_move  # 人类是黑方

        return is_human

    def TranslateMove(self, move):
        if not move or len(str(move)) != 4 or not str(move).isdigit():
            return "invalid"
        try:
            return chr(int(move[0]) + ord('a')) + str(move[1]) + chr(int(move[2]) + ord('a')) + str(move[3])
        except (ValueError, IndexError):
            return "invalid"

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
        pygame.display.set_caption("天衍象棋 发际线总和我作队 中国人民公安大学")

        try:
            icon = load_image('RK.gif', 'Piece')
            pygame.display.set_icon(icon)
        except:
            pass

        bgdtile = load_image('Board.GIF')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.height])
        board_background.blit(bgdtile, (0, 0))

        border_rect = pygame.Rect(0, 0, self.width, self.height)

        for i in range(4, 0, -1):
            border_width = i
            alpha = 40 + i * 20
            border_color = (*self.colors['accent'][:3], alpha)

            temp_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, border_color,
                             pygame.Rect(border_width - 1, border_width - 1,
                                         self.width - 2 * (border_width - 1),
                                         self.height - 2 * (border_width - 1)),
                             border_width)
            board_background.blit(temp_surface, (0, 0))

        # 添加高光边框
        highlight_rect = pygame.Rect(1, 1, self.width - 2, self.height - 2)
        pygame.draw.rect(board_background, (*self.colors['glass'][:3], 60), highlight_rect, 1)

        # 创建信息面板背景 - 使用高级渐变色
        widget_background = pygame.Surface([self.screen_width, self.screen_height - self.height])
        widget_rect = pygame.Rect(0, 0, self.screen_width, self.screen_height - self.height)
        self.draw_premium_gradient_rect(widget_background, widget_rect,
                                        self.colors['bg_primary'], self.colors['bg_secondary'],
                                        self.colors['bg_tertiary'])

        # 创建右侧信息面板背景 - 更豪华的渐变
        right_panel_width = self.screen_width - self.width
        right_panel_background = pygame.Surface([right_panel_width, self.height])
        right_panel_rect = pygame.Rect(0, 0, right_panel_width, self.height)
        self.draw_premium_gradient_rect(right_panel_background, right_panel_rect,
                                        self.colors['bg_panel'], self.colors['bg_elevated'],
                                        self.colors['bg_panel_hover'])

        screen.blit(board_background, (0, 0))
        screen.blit(right_panel_background, (self.width, 0))
        screen.blit(widget_background, (0, self.height))

        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w, self.chessman_h)
        return screen, board_background, widget_background

    def start(self, human_first=None):
        # 如果没有指定先手，则显示选择界面
        if human_first is None:
            human_first = self.choose_first_player()

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
        else:
            self.current_timer = 'red'  # AI是红方先手
            self.last_move_time = time.time()

        ai_worker = Thread(target=self.ai_move, name="ai_worker")
        ai_worker.daemon = True
        ai_worker.start()
        sleep(1)

        # 确保历史记录初始化
        if len(self.chessman_history) == 0:
            self.chessman_history.append(copy.deepcopy(self.env.board.chessmans))
            self.chessman_hash_history.append((copy.deepcopy(self.env.board.chessmans_hash)))
            self.record_history.append(copy.deepcopy(self.env.board.record))
            self.turn_history.append(copy.deepcopy(self.env.board.turns))

        def undo_move():
            if (len(self.chessman_history) > (2 if not self.human_move_first else 1)):
                logger.info("🔄 执行悔棋操作，回退2步")
                # 记录当前状态用于日志
                current_moves = len([m for m in self.env.board.record.split('\n') if m.strip()])

                # 重置AI状态和搜索树
                if hasattr(self, 'ai') and self.ai:
                    logger.info("🧠 清理AI搜索树和状态")
                    try:
                        if hasattr(self.ai, 'tree'):
                            tree_size = len(self.ai.tree)
                            self.ai.tree.clear()  # 清空搜索树
                            logger.info(f"✅ AI搜索树已清理，原节点数: {tree_size}")
                        if hasattr(self.ai, 'search_results'):
                            self.ai.search_results.clear()
                        if hasattr(self.ai, 'debug'):
                            self.ai.debug.clear()
                        self.mcts_moves = {}  # 清空 MCTS 结果
                        logger.info("🔄 AI状态重置完成")
                    except Exception as e:
                        logger.warning(f"⚠️ AI状态重置时出现警告: {str(e)}")
                        # 继续执行悔棋，不让这个错误阻止悔棋操作

                for i in range(4):
                    self.history.pop()
                for i in range(2):
                    self.moves_history.pop()
                self.chessman_history.pop()
                self.record_history.pop()
                self.turn_history.pop()
                self.chessman_hash_history.pop()

                # 重置棋盘状态
                self.env.board.Set_chessmans(copy.deepcopy(self.chessman_history[-1]))
                self.env.board.Set_chessmans_hash(copy.deepcopy(self.chessman_hash_history[-1]))
                self.env.board.turns = self.turn_history[-1]
                self.env.board.record = self.record_history[-1]

                # 重新计算可移动列表
                self.env.board.calc_chessmans_moving_list()

                # 重置环境状态
                self.env.num_halfmoves = self.env.board.turns
                # 注意：不能直接设置 self.env.done，它是基于 winner 的计算属性
                # 确保winner状态被正确重置（通过棋盘状态恢复）
                if hasattr(self.env, 'winner'):
                    self.env.winner = None  # 重置胜负状态
                if hasattr(self.env.board, 'winner'):
                    self.env.board.winner = None  # 重置棋盘胜负状态
                logger.info("🔄 环境状态已重置")

                # 重新创建棋子精灵
                self.chessmans.empty()
                for chess in self.env.board.chessmans_hash.values():
                    if chess.is_red:
                        if isinstance(chess, Rook):
                            images = load_images("RR.gif", "RRS.gif")
                        elif isinstance(chess, Cannon):
                            images = load_images("RC.gif", "RCS.gif")
                        elif isinstance(chess, Knight):
                            images = load_images("RN.gif", "RNS.gif")
                        elif isinstance(chess, King):
                            images = load_images("RK.gif", "RKS.gif")
                        elif isinstance(chess, Elephant):
                            images = load_images("RB.gif", "RBS.gif")
                        elif isinstance(chess, Mandarin):
                            images = load_images("RA.gif", "RAS.gif")
                        else:
                            images = load_images("RP.gif", "RPS.gif")
                    else:
                        if isinstance(chess, Rook):
                            images = load_images("BR.gif", "BRS.gif")
                        elif isinstance(chess, Cannon):
                            images = load_images("BC.gif", "BCS.gif")
                        elif isinstance(chess, Knight):
                            images = load_images("BN.gif", "BNS.gif")
                        elif isinstance(chess, King):
                            images = load_images("BK.gif", "BKS.gif")
                        elif isinstance(chess, Elephant):
                            images = load_images("BB.gif", "BBS.gif")
                        elif isinstance(chess, Mandarin):
                            images = load_images("BA.gif", "BAS.gif")
                        else:
                            images = load_images("BP.gif", "BPS.gif")
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

                # 悔棋成功日志
                remaining_moves = len([m for m in self.env.board.record.split('\n') if m.strip()])
                logger.info(f"✅ 悔棋成功！从第{current_moves}步回退到第{remaining_moves}步")

                # 重置AI历史记录，确保与主游戏状态同步
                if hasattr(self, 'ai') and self.ai:
                    # 重新同步history列表，这很重要！
                    logger.info("🔄 重新同步AI历史记录")
                    # AI线程会使用这个history，必须保持一致

                # 重置计时器到人类玩家回合
                if self.human_move_first:
                    self.current_timer = 'red'  # 人类是红方
                else:
                    self.current_timer = 'black'  # 人类是黑方
                self.last_move_time = time.time()

                logger.info(f"🔄 计时器已重置到人类玩家回合: {self.current_timer}")

                # 清除当前选中的棋子状态
                logger.info("🎯 重置棋子选择状态")
                # 这个变量在主循环中定义，这里无法直接访问，但我们可以确保棋子精灵状态正确
                for sprite in self.chessmans:
                    if hasattr(sprite, 'is_selected'):
                        sprite.is_selected = False

                # 确保环境状态与当前应该轮到的玩家一致
                logger.info(
                    f"🔄 当前轮到: {'人类' if self.is_human_turn() else 'AI'} ({'红方' if self.env.red_to_move else '黑方'})")
            else:
                logger.warning("⚠️ 不存在可以悔棋的历史局面!")
                print("不存在可以悔棋的历史局面!")
                self.set_operation_feedback("⚠️ 无法悔棋！", 'warning')

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
                    result = self.env.board.get_result_string()
                    red_team = "玩家"
                    black_team = "AI"
                    self.env.board.save_record("", "", red_team, black_team, result)
                    logger.info(f"游戏被用户关闭，棋谱已保存")
                    pygame.quit()
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    pass
                elif event.type == MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if self.hittest(mouse_x, mouse_y, (20, self.height + 20, 120, 45)):
                        if self.is_human_turn():
                            logger.info("🔄 玩家请求悔棋")
                            print("悔棋!\n")
                            self.set_operation_feedback("🔄 正在执行悔棋...", 'info')
                            undo_move()
                            # 悔棋后清除当前选中的棋子
                            current_chessman = None
                            self.set_operation_feedback("✅ 悔棋成功！", 'success')
                        else:
                            logger.info("⚠️ AI回合时无法悔棋")
                            print("AI思考中，无法悔棋！")
                            self.set_operation_feedback("⚠️ AI回合无法悔棋！", 'warning')
                    elif self.hittest(mouse_x, mouse_y, (160, self.height + 20, 150, 45)):
                        logger.info("📄 玩家请求保存棋谱")
                        print("保存棋谱!")
                        self.set_operation_feedback("📄 正在保存棋谱...", 'info')
                        self.env.board.print_record()
                        try:
                            result = self.env.board.get_result_string()
                            red_team = "玩家"
                            black_team = "AI"
                            self.env.board.save_record("", "", red_team, black_team, result)
                            logger.info(f"✅ 棋谱已保存")
                            print(f"棋谱已保存")
                            self.set_operation_feedback(f"✅ 棋谱已保存！", 'success')
                        except Exception as e:
                            logger.error(f"❌ 棋谱保存失败: {str(e)}")
                            print(f"棋谱保存失败: {str(e)}")
                            self.set_operation_feedback("❌ 棋谱保存失败！", 'error')
                    elif self.is_human_turn():  # 使用新的判断函数
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
                                            # # 检查三次重复局面
                                            # if self.check_threefold_repetition():
                                            #     break  # 如果判定和棋，跳出循环
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

        # 检查胜负是否真正确定
        if self.is_winner_definitive():
            # 记录游戏结束时间
            self.game_stats['current_game_end'] = time.time()

            self.env.board.print_record()
            result = self.env.board.get_result_string()
            red_team = "玩家"
            black_team = "AI"
            self.env.board.save_record("", "", red_team, black_team, result)

            # 显示胜利界面，让用户选择下一步行动
            self.show_game_over_dialog(screen, widget_background)
        else:
            # 胜负不确定，重置winner状态，让游戏继续
            logger.info("胜负不确定，重置winner状态，游戏继续...")
            self.env.winner = None
            self.env.board.winner = None
            self.set_operation_feedback("🔄 检测到异常状态，游戏继续...", 'info')
            
            # 重新启动游戏循环
            self.continue_game_after_uncertainty(screen, widget_background)

    def ai_move(self):
        ai_move_first = not self.human_move_first
        self.history = [self.env.get_state()]
        # self.gui_history.append(copy.deepcopy(self.chessmans))
        no_act = None
        while not self.env.done:
            if ai_move_first == self.env.red_to_move:
                # 添加短暂延迟，让主线程有时间处理悔棋等操作
                sleep(0.1)

                # 检查游戏是否仍在进行中
                if self.env.done or self.env.board.is_end():
                    break

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
                            if senv.will_check_or_catch(state, self.history[i + 1]):
                                no_act.append(self.history[i + 1])
                            else:
                                free_move[state] += 1
                                if free_move[state] >= 2:
                                # 作和棋处理
                                   self.env.winner = Winner.draw
                                   self.env.board.winner = Winner.draw
                                   break
                    if no_act:
                        logger.debug(f"no_act = {no_act}")
                BookResult = []
                if not self.config.resource.Use_Book:
                    self.book_msg = "未启用历史局面缓存..."
                if self.config.resource.Use_Book and (
                        self.config.resource.Out_Book_Step == -1 or self.env.board.turns <= self.config.resource.Out_Book_Step):
                    BookResult = self.bookhandler.query(BookHandler.fixFen(self.env.get_state()), ai_move_first)
                    BookResult.sort(key=cmp_to_key(Bookcmp), reverse=1)
                    if len(BookResult) > 0:
                        self.book_msg = '命中历史局面!'
                    else:
                        self.book_msg = '未命中历史局面!'

                action = None
                self.BsetMove = []
                if not self.config.resource.Use_EngineHelp:
                    action, policy = self.ai.action(state, self.env.num_halfmoves, no_act)
                    if not self.env.red_to_move:
                        action = flip_move(action)
                if self.config.resource.Use_Book and len(BookResult) > 0:
                    for i in range(min(3, len(BookResult))):
                        self.BsetMove.append(BookResult[i].move)
                    self.info_msg = '从历史局面获取最优走法:'
                    print("使用开局库:", str(BookResult[0].move))
                    if BookResult[0].move and len(str(BookResult[0].move)) >= 4:
                        x0, y0, x1, y1 = BookHandler.Move2Point(BookResult[0].move)
                        action = str(x0) + str(y0) + str(x1) + str(y1)
                        self.history.append(action)
                        self.moves_history.append(self.TranslateMove(action))
                    else:
                        return
                elif self.config.resource.Use_EngineHelp:
                    action = Engine_Manager.get_uci_move(self.config.resource.engine_path,
                                                         self.moves_history,
                                                         self.config.resource.EngineSearchThreads,
                                                         ai_move_first,
                                                         self.config.resource.EngineSearchTime + random.randint(5, 10))
                    self.info_msg = "AlphaZero搜索的最优走法:"
                    self.BsetMove.append(action)

                    print("使用AI搜索: ", action)
                    if action and action != "(none)" and len(action) >= 4:
                        x0, y0, x1, y1 = BookHandler.Move2Point(action)
                        action = str(x0) + str(y0) + str(x1) + str(y1)
                        self.history.append(action)
                        self.moves_history.append(self.TranslateMove(action))
                    else:
                        return
                else:
                    self.BsetMove.append(action)
                    self.info_msg = "MCTS搜索的最优走法:"
                    print("使用MCTS搜索: ", action)
                    if action and len(str(action)) == 4 and str(action).isdigit():
                        self.history.append(action)
                        self.moves_history.append(self.TranslateMove(action))
                    else:
                        return
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

                if not action or len(str(action)) != 4 or not str(action).isdigit():
                    return
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                if chessman_sprite is None:
                    logger.error(f"❌ AI移动失败：找不到起始位置({x0},{y0})的棋子")
                    return

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
        # 重新绘制高级渐变背景
        widget_rect = pygame.Rect(0, 0, self.screen_width, self.screen_height - self.height)
        self.draw_premium_gradient_rect(widget_background, widget_rect,
                                        self.colors['bg_primary'], self.colors['bg_secondary'],
                                        self.colors['bg_tertiary'])

        # 创建右侧面板
        right_panel_width = self.screen_width - self.width
        right_panel_surface = pygame.Surface([right_panel_width, self.height])
        right_panel_rect = pygame.Rect(0, 0, right_panel_width, self.height)
        self.draw_premium_gradient_rect(right_panel_surface, right_panel_rect,
                                        self.colors['bg_panel'], self.colors['bg_elevated'],
                                        self.colors['bg_panel_hover'])

        # 在右侧面板绘制计时器
        timer_width = 150
        timer_height = 45
        margin = 15

        red_timer_rect = pygame.Rect(margin, margin, timer_width, timer_height)
        black_timer_rect = pygame.Rect(margin, margin + timer_height + 8, timer_width, timer_height)

        is_red_active = self.current_timer == 'red'
        is_black_active = self.current_timer == 'black'

        self.draw_timer_display(right_panel_surface, red_timer_rect, self.red_time_left, "红方", is_red_active)
        self.draw_timer_display(right_panel_surface, black_timer_rect, self.black_time_left, "黑方", is_black_active)

        # 游戏统计面板
        stats_panel_y = margin + (timer_height + 8) * 2 + 10
        stats_rect = pygame.Rect(margin, stats_panel_y, right_panel_width - 2 * margin, 85)
        self.draw_game_stats_panel(right_panel_surface, stats_rect)

        # 在右侧面板绘制AI信息面板
        # info_panel_y = stats_panel_y + 95
        # if self.config.resource.Use_EngineHelp:
        #     info_rect = pygame.Rect(margin, info_panel_y, right_panel_width - 2 * margin, 100)
        #     info_lines = []
        #     if self.book_msg:
        #         info_lines.append(self.book_msg)
        #     if self.info_msg:
        #         info_lines.append(self.info_msg)
        #     for i, move in enumerate(self.BsetMove[:2]):
        #         info_lines.append(f"{i + 1}. {move}")
        #     if self.BsetMove:
        #         info_lines.append(f"执行: {self.BsetMove[0]}")
        #
        #     self.draw_info_panel(right_panel_surface, info_rect, "🧠 AI 分析", info_lines)
        # else:
        #     # 绘制评估信息
        #     info_rect = pygame.Rect(margin, info_panel_y, right_panel_width - 2 * margin, 100)
        #     self.draw_evaluation_panel(right_panel_surface, info_rect)

        # 局面评估面板
        eval_panel_y = stats_panel_y + 110
        eval_rect = pygame.Rect(margin, eval_panel_y, right_panel_width - 2 * margin, 80)
        self.draw_position_analysis_panel(right_panel_surface, eval_rect)

        # 在右侧面板绘制对局记录
        record_panel_y = eval_panel_y + 90
        record_rect = pygame.Rect(margin, record_panel_y, right_panel_width - 2 * margin,
                                  self.height - record_panel_y - margin)
        self.draw_right_panel_records(right_panel_surface, record_rect)

        # 在底部面板绘制控制按钮
        button_y = 20
        button_width = 80  # 从100再缩小到80
        button_height = 35  # 从40再缩小到35
        button_spacing = 12  # 从15再缩小到12

        # 悔棋按钮 - 使用与右侧面板一致的深色风格
        undo_rect = pygame.Rect(20, button_y, button_width, button_height)
        is_human_turn = self.is_human_turn()
        undo_button_text = "🔄 悔棋" if is_human_turn else "⏸️ AI中"
        self.draw_info_panel_style_button(widget_background, undo_rect, undo_button_text, 12,
                                          active=is_human_turn)  # 字体从14调到12

        # 打印棋谱按钮 - 使用与右侧面板一致的深色风格
        print_rect = pygame.Rect(20 + button_width + button_spacing, button_y, button_width + 20,
                                 button_height)  # 宽度从+25调到+20
        self.draw_info_panel_style_button(widget_background, print_rect, "📄 保存棋谱", 12)  # 字体从14调到12

        # 状态信息面板 - 进一步调整位置和宽度
        status_rect = pygame.Rect(240, button_y, 440, button_height)  # 向左移动20px，宽度增加20px
        now = time.strftime('%H:%M:%S')
        current_player = "红方" if self.env.red_to_move else "黑方"

        # 计算当前游戏时间
        if not self.game_stats.get('current_game_start'):
            self.game_stats['current_game_start'] = time.time()

        game_time = time.time() - self.game_stats['current_game_start']
        status_lines = [
            f"🕰️ {now} | 🎯 {current_player}",
            f"🎮 对局时间: {int(game_time // 60):02d}:{int(game_time % 60):02d}"  # 恢复完整文字
        ]
        self.draw_info_panel(widget_background, status_rect, None, status_lines, smaller_font=True)

        # 操作反馈面板
        self.draw_operation_feedback_panel(widget_background)

        # 绘制详细对局记录 - 显示更多历史步数，使用更紧凑的格式
        self.draw_detailed_records(screen, widget_background)

        # 将右侧面板绘制到屏幕
        screen.blit(right_panel_surface, (self.width, 0))
        screen.blit(widget_background, (0, self.height))

    def draw_undo_button(self, screen, widget_background):
        """悔棋按钮已在draw_widget中重新实现"""
        pass

    def draw_records(self, screen, widget_background):
        """绘制对局记录 - 美化版本"""
        record_rect = pygame.Rect(10, 135, 500, 0)  # 动态高度

        # 准备记录内容
        record_lines = []
        if hasattr(self.env.board, 'record') and self.env.board.record:
            moves = self.env.board.record.strip().split('\n')
            for i, move in enumerate(moves[-8:], 1):  # 显示最近8步
                if move.strip():
                    record_lines.append(f"{len(moves) - 8 + i}. {move.strip()}")

        if not record_lines:
            record_lines = ["等待开局..."]

        # 动态计算面板高度
        record_rect.height = max(60, len(record_lines) * 18 + 40)

        self.draw_info_panel(widget_background, record_rect, "📃 对局记录", record_lines)

    def draw_right_panel_records(self, surface, rect):
        """绘制右侧面板的详细对局记录"""
        record_lines = []
        if hasattr(self.env.board, 'record') and self.env.board.record:
            moves = self.env.board.record.strip().split('\n')
            for i, move in enumerate(moves[-15:], 1):  # 显示最近15步
                if move.strip():
                    step_num = len(moves) - 15 + i
                    record_lines.append(f"{step_num}. {move.strip()}")

        if not record_lines:
            record_lines = ["等待开局..."]

        self.draw_info_panel(surface, rect, "📜 棋步记录", record_lines)

    def draw_detailed_records(self, screen, widget_background):
        """绘制详细的对局记录，使用更紧凑的多列布局"""
        record_rect = pygame.Rect(20, 75, 660, 190)

        record_lines = []
        if hasattr(self.env.board, 'record') and self.env.board.record:
            moves = self.env.board.record.strip().split('\n')
            # 使用三列布局显示更多棋步，但控制每行长度
            for i in range(0, len(moves[-18:]), 3):  # 显示最近18步，每行3步
                line_moves = moves[-18 + i:min(-18 + i + 3, len(moves))]
                formatted_line = ""
                for j, move in enumerate(line_moves):
                    if move.strip():
                        step_num = len(moves) - 18 + i + j + 1
                        # 限制单个棋步显示长度，避免超出框架
                        move_text = move.strip()
                        if len(move_text) > 8:
                            move_text = move_text[:8] + ".."
                        formatted_line += f"{step_num}.{move_text:<10} "
                if formatted_line:
                    record_lines.append(formatted_line.rstrip())

        if not record_lines:
            record_lines = ["等待开局..."]

        # 添加总结信息
        if hasattr(self.env.board, 'record') and self.env.board.record:
            total_moves = len([m for m in self.env.board.record.split('\n') if m.strip()])
            current_player = '红方' if self.env.red_to_move else '黑方'
            record_lines.append("")
            record_lines.append(f"📊 总步数: {total_moves} | 当前: {current_player}")

        self.draw_info_panel(widget_background, record_rect, "📋 完整棋谱", record_lines)

    def draw_game_stats_panel(self, surface, rect):
        """绘制游戏统计信息面板"""
        stats_lines = []

        # 确保游戏开始时间已设置
        if not self.game_stats.get('current_game_start'):
            self.game_stats['current_game_start'] = time.time()

        current_game_time = time.time() - self.game_stats['current_game_start']
        current_moves = len(self.env.board.record.split('\n')) if hasattr(self.env.board,
                                                                          'record') and self.env.board.record else 0

        stats_lines.append(f"🎮 当前局: 第{current_moves}步")
        stats_lines.append(f"⏱️ 用时: {int(current_game_time // 60):02d}:{int(current_game_time % 60):02d}")
        stats_lines.append(f"🏆 历史: {self.game_stats['player_wins']}胜 {self.game_stats['ai_wins']}败")

        self.draw_info_panel(surface, rect, "📊 对局统计", stats_lines)

    def draw_position_analysis_panel(self, surface, rect):
        """绘制局面分析面板"""
        analysis_lines = []

        # 计算棋子价值
        red_value, black_value = self.calculate_material_balance()
        material_diff = red_value - black_value

        if material_diff > 0:
            analysis_lines.append(f"⚖️ 红方领先 {material_diff:.1f} 分")
        elif material_diff < 0:
            analysis_lines.append(f"⚖️ 黑方领先 {abs(material_diff):.1f} 分")
        else:
            analysis_lines.append("⚖️ 局面均衡")

        # 显示当前轮到谁
        current_player = "红方" if self.env.red_to_move else "黑方"
        analysis_lines.append(f"🎯 轮到: {current_player}")

        # 显示危险度
        if hasattr(self, 'nn_value') and self.nn_value:
            if self.nn_value > 0.3:
                analysis_lines.append("🔥 红方优势明显")
            elif self.nn_value < -0.3:
                analysis_lines.append("❄️ 黑方优势明显")
            else:
                analysis_lines.append("⚡ 形势相当")

        self.draw_info_panel(surface, rect, "🔍 局面分析", analysis_lines)

    def calculate_material_balance(self):
        """计算双方棋子价值"""
        piece_values = {
            'King': 0,  # 将帅无价值（游戏结束条件）
            'Mandarin': 2,  # 士
            'Elephant': 2,  # 象
            'Knight': 4,  # 马
            'Rook': 9,  # 车
            'Cannon': 4.5,  # 炮
            'Pawn': 1  # 兵/卒
        }

        red_total = 0
        black_total = 0

        if hasattr(self.env.board, 'chessmans_hash'):
            for chess in self.env.board.chessmans_hash.values():
                piece_name = chess.__class__.__name__
                value = piece_values.get(piece_name, 0)

                if chess.is_red:
                    red_total += value
                else:
                    black_total += value

        return red_total, black_total

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

    def draw_label(self, screen, widget_background, text, y, font_size, x=None, color=None):
        """绘制文本标签 - 支持新主题"""
        if not text:
            return

        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, font_size)

        # 使用主题颜色
        text_color = color if color else self.colors['text_primary']
        label = font.render(str(text), True, text_color)

        t_rect = label.get_rect()
        t_rect.y = y
        if x is not None:
            t_rect.x = x
        else:
            t_rect.centerx = (self.screen_width - self.width) / 2

        widget_background.blit(label, t_rect)
        screen.blit(widget_background, (0, self.height))

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def draw_premium_gradient_rect(self, surface, rect, color1, color2, color3=None, vertical=True):
        """绘制豪华三色渐变矩形"""
        if color3 is None:
            color3 = color2

        if vertical:
            for y in range(rect.height):
                ratio = y / rect.height

                # 三阶段渐变：0-0.4使用color1到color2，0.4-0.7使用color2，0.7-1.0使用color2到color3
                if ratio <= 0.4:
                    # 第一阶段
                    stage_ratio = ratio / 0.4
                    smooth_ratio = stage_ratio * stage_ratio * (3.0 - 2.0 * stage_ratio)
                    r = int(color1[0] * (1 - smooth_ratio) + color2[0] * smooth_ratio)
                    g = int(color1[1] * (1 - smooth_ratio) + color2[1] * smooth_ratio)
                    b = int(color1[2] * (1 - smooth_ratio) + color2[2] * smooth_ratio)
                elif ratio <= 0.7:
                    # 第二阶段（稳定色）
                    r, g, b = color2
                else:
                    # 第三阶段
                    stage_ratio = (ratio - 0.7) / 0.3
                    smooth_ratio = stage_ratio * stage_ratio * (3.0 - 2.0 * stage_ratio)
                    r = int(color2[0] * (1 - smooth_ratio) + color3[0] * smooth_ratio)
                    g = int(color2[1] * (1 - smooth_ratio) + color3[1] * smooth_ratio)
                    b = int(color2[2] * (1 - smooth_ratio) + color3[2] * smooth_ratio)

                pygame.draw.line(surface, (r, g, b),
                                 (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))
        else:
            for x in range(rect.width):
                ratio = x / rect.width

                if ratio <= 0.4:
                    stage_ratio = ratio / 0.4
                    smooth_ratio = stage_ratio * stage_ratio * (3.0 - 2.0 * stage_ratio)
                    r = int(color1[0] * (1 - smooth_ratio) + color2[0] * smooth_ratio)
                    g = int(color1[1] * (1 - smooth_ratio) + color2[1] * smooth_ratio)
                    b = int(color1[2] * (1 - smooth_ratio) + color2[2] * smooth_ratio)
                elif ratio <= 0.7:
                    r, g, b = color2
                else:
                    stage_ratio = (ratio - 0.7) / 0.3
                    smooth_ratio = stage_ratio * stage_ratio * (3.0 - 2.0 * stage_ratio)
                    r = int(color2[0] * (1 - smooth_ratio) + color3[0] * smooth_ratio)
                    g = int(color2[1] * (1 - smooth_ratio) + color3[1] * smooth_ratio)
                    b = int(color2[2] * (1 - smooth_ratio) + color3[2] * smooth_ratio)

                pygame.draw.line(surface, (r, g, b),
                                 (rect.x + x, rect.y), (rect.x + x, rect.y + rect.height))

    def draw_gradient_rect(self, surface, rect, color1, color2, vertical=True):
        """绘制增强的渐变矩形"""
        if vertical:
            for y in range(rect.height):
                ratio = y / rect.height
                # 使用更平滑的渐变曲线
                smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)  # 平滑插值
                r = int(color1[0] * (1 - smooth_ratio) + color2[0] * smooth_ratio)
                g = int(color1[1] * (1 - smooth_ratio) + color2[1] * smooth_ratio)
                b = int(color1[2] * (1 - smooth_ratio) + color2[2] * smooth_ratio)
                pygame.draw.line(surface, (r, g, b),
                                 (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))
        else:
            for x in range(rect.width):
                ratio = x / rect.width
                smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                r = int(color1[0] * (1 - smooth_ratio) + color2[0] * smooth_ratio)
                g = int(color1[1] * (1 - smooth_ratio) + color2[1] * smooth_ratio)
                b = int(color1[2] * (1 - smooth_ratio) + color2[2] * smooth_ratio)
                pygame.draw.line(surface, (r, g, b),
                                 (rect.x + x, rect.y), (rect.x + x, rect.y + rect.height))

    def draw_rounded_rect(self, surface, rect, color, radius=10, border_color=None, border_width=0):
        """绘制圆角矩形"""
        if radius > min(rect.width, rect.height) // 2:
            radius = min(rect.width, rect.height) // 2

        # 绘制主体矩形
        inner_rect = pygame.Rect(rect.x + radius, rect.y, rect.width - 2 * radius, rect.height)
        pygame.draw.rect(surface, color, inner_rect)

        inner_rect = pygame.Rect(rect.x, rect.y + radius, rect.width, rect.height - 2 * radius)
        pygame.draw.rect(surface, color, inner_rect)

        # 绘制四个圆角
        pygame.draw.circle(surface, color, (rect.x + radius, rect.y + radius), radius)
        pygame.draw.circle(surface, color, (rect.x + rect.width - radius, rect.y + radius), radius)
        pygame.draw.circle(surface, color, (rect.x + radius, rect.y + rect.height - radius), radius)
        pygame.draw.circle(surface, color, (rect.x + rect.width - radius, rect.y + rect.height - radius), radius)

        # 绘制边框
        if border_color and border_width > 0:
            pygame.draw.rect(surface, border_color, rect, border_width, radius)

    def draw_info_panel_style_button(self, surface, rect, text, font_size=16, active=True, hover=False):
        """绘制与信息面板一致风格的按钮"""
        # 绘制深度阴影
        for i in range(4, 0, -1):
            shadow_rect = pygame.Rect(rect.x + i, rect.y + i, rect.width, rect.height)
            alpha = 25 - i * 5
            self.draw_rounded_rect_with_alpha(surface, shadow_rect, (0, 0, 0, alpha), 10)

        # 绘制面板背景 - 使用与信息面板相同的渐变
        panel_surface = pygame.Surface((rect.width, rect.height))
        panel_rect = pygame.Rect(0, 0, rect.width, rect.height)

        if active:
            # 激活状态使用稍亮的背景
            self.draw_premium_gradient_rect(panel_surface, panel_rect,
                                            self.colors['bg_elevated'],
                                            self.colors['bg_panel'],
                                            self.colors['bg_panel_hover'])
        else:
            # 非激活状态使用更暗的背景
            self.draw_premium_gradient_rect(panel_surface, panel_rect,
                                            self.colors['bg_tertiary'],
                                            self.colors['bg_secondary'],
                                            self.colors['bg_tertiary'])
        surface.blit(panel_surface, (rect.x, rect.y))

        # 绘制玻璃效果高光
        highlight_rect = pygame.Rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height // 4)
        self.draw_rounded_rect_with_alpha(surface, highlight_rect, self.colors['glass'], 8)

        # 绘制边框（多层）
        border_color = self.colors['accent'] if active else self.colors['border']
        self.draw_rounded_rect(surface, rect, (0, 0, 0, 0), 10, border_color, 1)
        inner_border_rect = pygame.Rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2)
        self.draw_rounded_rect(surface, inner_border_rect, (0, 0, 0, 0), 9, self.colors['border_light'], 1)

        # 绘制文字
        font = pygame.font.Font(self.config.resource.font_path, font_size)

        # 文字阴影
        text_shadow = font.render(text, True, (0, 0, 0, 100))
        shadow_rect = text_shadow.get_rect(center=(rect.centerx + 1, rect.centery + 1))
        surface.blit(text_shadow, shadow_rect)

        # 主文字
        text_color = self.colors['text_primary'] if active else self.colors['text_muted']
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def draw_enhanced_button(self, surface, rect, text, font_size=18, active=True, hover=False):
        """绘制增强可见性按钮 - 简洁清晰设计"""
        # 选择简洁的颜色方案
        if active:
            bg_color = (255, 255, 255)  # 纯白背景
            text_color = (30, 30, 30)  # 纯黑文字
            border_color = self.colors['accent']  # 金黄色边框
        else:
            bg_color = (200, 200, 200)  # 浅灰背景
            text_color = (80, 80, 80)  # 深灰文字
            border_color = (120, 120, 120)  # 灰色边框

        # 绘制简单阴影
        shadow_rect = pygame.Rect(rect.x + 2, rect.y + 2, rect.width, rect.height)
        pygame.draw.rect(surface, (0, 0, 0, 60), shadow_rect, border_radius=8)

        # 绘制主按钮背景
        pygame.draw.rect(surface, bg_color, rect, border_radius=8)

        # 绘制边框
        pygame.draw.rect(surface, border_color, rect, 2, border_radius=8)

        # 绘制文字 - 简单清晰
        font = pygame.font.Font(self.config.resource.font_path, font_size)
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def draw_light_button(self, surface, rect, text, font_size=16, active=True, hover=False):
        """绘制浅色按钮 - 白色背景，深色文字"""
        # 选择颜色方案
        if active:
            bg_color = (248, 250, 252)  # 亮白色背景
            text_color = (31, 41, 55)  # 深色文字
            border_color = self.colors['accent']
            shadow_color = (*self.colors['accent'][:3], 60)
        else:
            bg_color = (203, 213, 225)  # 灰白色背景
            text_color = (100, 116, 139)  # 灰色文字
            border_color = self.colors['border']
            shadow_color = self.colors['shadow_light']

        # 绘制多层深度阴影效果
        for i in range(4, 0, -1):
            shadow_rect = pygame.Rect(rect.x + i, rect.y + i, rect.width, rect.height)
            alpha = 25 - i * 5
            self.draw_rounded_rect_with_alpha(surface, shadow_rect, (0, 0, 0, alpha), 12)

        # 绘制发光效果（仅激活状态）
        if active:
            glow_rect = pygame.Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2)
            self.draw_rounded_rect_with_alpha(surface, glow_rect, shadow_color, 13)

        # 主按钮背景 - 使用浅色渐变
        if active:
            # 创建按钮表面用于渐变
            button_surface = pygame.Surface((rect.width, rect.height))
            button_rect = pygame.Rect(0, 0, rect.width, rect.height)
            lighter_bg = tuple(min(255, c + 15) for c in bg_color)
            darker_bg = tuple(max(0, c - 15) for c in bg_color)
            self.draw_premium_gradient_rect(button_surface, button_rect,
                                            lighter_bg, bg_color, darker_bg)
            surface.blit(button_surface, (rect.x, rect.y))
        else:
            pygame.draw.rect(surface, bg_color, rect, border_radius=12)

        # 绘制边框
        self.draw_rounded_rect(surface, rect, (0, 0, 0, 0), 12, border_color, 2)

        # 顶部高光
        highlight_rect = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height // 4)
        self.draw_rounded_rect_with_alpha(surface, highlight_rect, (255, 255, 255, 80), 8)

        # 绘制文字（带阴影）
        font = pygame.font.Font(self.config.resource.font_path, font_size)

        # 浅色阴影
        shadow_surface = font.render(text, True, (0, 0, 0, 40))
        shadow_rect = shadow_surface.get_rect(center=(rect.centerx + 1, rect.centery + 1))
        surface.blit(shadow_surface, shadow_rect)

        # 主文字
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def draw_modern_button(self, surface, rect, text, font_size=16, active=True, hover=False):
        """绘制现代化按钮 - 豪华版"""
        # 选择颜色方案
        if active:
            bg_color = self.colors['accent'] if not hover else self.colors['accent_hover']
            text_color = self.colors['bg_primary']
            border_color = self.colors['border_accent']
            shadow_color = (*self.colors['accent'][:3], 60)
        else:
            bg_color = self.colors['bg_tertiary']
            text_color = self.colors['text_muted']
            border_color = self.colors['border']
            shadow_color = self.colors['shadow_light']

        # 绘制多层深度阴影效果
        for i in range(5, 0, -1):
            shadow_rect = pygame.Rect(rect.x + i, rect.y + i, rect.width, rect.height)
            alpha = 20 - i * 3
            self.draw_rounded_rect_with_alpha(surface, shadow_rect, (0, 0, 0, alpha), 12)

        # 绘制发光效果（仅激活状态）
        if active:
            glow_rect = pygame.Rect(rect.x - 2, rect.y - 2, rect.width + 4, rect.height + 4)
            self.draw_rounded_rect_with_alpha(surface, glow_rect, shadow_color, 14)

        # 主按钮背景 - 使用高级渐变
        if active:
            # 创建按钮表面用于渐变
            button_surface = pygame.Surface((rect.width, rect.height))
            button_rect = pygame.Rect(0, 0, rect.width, rect.height)
            self.draw_premium_gradient_rect(button_surface, button_rect,
                                            bg_color,
                                            tuple(max(0, c - 40) for c in bg_color),
                                            tuple(max(0, c - 20) for c in bg_color))
            surface.blit(button_surface, (rect.x, rect.y))
        else:
            pygame.draw.rect(surface, bg_color, rect, border_radius=12)

        # 绘制边框和高光
        self.draw_rounded_rect(surface, rect, (0, 0, 0, 0), 12, border_color, 2)

        # 顶部高光
        if active:
            highlight_rect = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height // 3)
            self.draw_rounded_rect_with_alpha(surface, highlight_rect, self.colors['highlight'], 8)

        # 绘制文字（带多层阴影）
        font = pygame.font.Font(self.config.resource.font_path, font_size)

        # 深层阴影
        for i in range(3, 0, -1):
            shadow_surface = font.render(text, True, (0, 0, 0, 80 - i * 20))
            shadow_rect = shadow_surface.get_rect(center=(rect.centerx + i, rect.centery + i))
            surface.blit(shadow_surface, shadow_rect)

        # 主文字
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def draw_rounded_rect_with_alpha(self, surface, rect, color, radius=10):
        """绘制带透明度的圆角矩形"""
        if len(color) == 4 and color[3] < 255:
            # 创建临时表面用于透明度
            temp_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            temp_surface = temp_surface.convert_alpha()
            self.draw_rounded_rect(temp_surface, pygame.Rect(0, 0, rect.width, rect.height), color, radius)
            surface.blit(temp_surface, (rect.x, rect.y))
        else:
            self.draw_rounded_rect(surface, rect, color[:3], radius)

    def draw_info_panel(self, surface, rect, title, content_lines, icon_color=None, smaller_font=False):
        """绘制信息面板 - 豪华版，支持深度视觉效果"""
        # 绘制深度阴影
        for i in range(4, 0, -1):
            shadow_rect = pygame.Rect(rect.x + i, rect.y + i, rect.width, rect.height)
            alpha = 25 - i * 5
            self.draw_rounded_rect_with_alpha(surface, shadow_rect, (0, 0, 0, alpha), 10)

        # 绘制面板背景 - 使用高级渐变
        panel_surface = pygame.Surface((rect.width, rect.height))
        panel_rect = pygame.Rect(0, 0, rect.width, rect.height)
        self.draw_premium_gradient_rect(panel_surface, panel_rect,
                                        self.colors['bg_elevated'],
                                        self.colors['bg_panel'],
                                        self.colors['bg_panel_hover'])
        surface.blit(panel_surface, (rect.x, rect.y))

        # 绘制玻璃效果高光
        highlight_rect = pygame.Rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height // 4)
        self.draw_rounded_rect_with_alpha(surface, highlight_rect, self.colors['glass'], 8)

        # 绘制边框（多层）
        self.draw_rounded_rect(surface, rect, (0, 0, 0, 0), 10, self.colors['border'], 1)
        inner_border_rect = pygame.Rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2)
        self.draw_rounded_rect(surface, inner_border_rect, (0, 0, 0, 0), 9, self.colors['border_light'], 1)

        y_offset = rect.y + 12
        max_width = rect.width - 24  # 增加边距

        # 绘制标题
        if title:
            title_font_size = 12 if smaller_font else 14
            font = pygame.font.Font(self.config.resource.font_path, title_font_size)

            # 标题阴影
            title_shadow = font.render(title, True, (0, 0, 0, 100))
            surface.blit(title_shadow, (rect.x + 13, y_offset + 1))

            # 主标题
            title_surface = font.render(title, True, self.colors['accent'])
            surface.blit(title_surface, (rect.x + 12, y_offset))
            y_offset += 25

        # 绘制内容 - 添加文本折行和长度限制
        content_font_size = 10 if smaller_font else 12
        font = pygame.font.Font(self.config.resource.font_path, content_font_size)
        for line in content_lines:
            if line and y_offset < rect.y + rect.height - 20:  # 确保不超出面板高度
                # 处理文本换行
                wrapped_lines = self.wrap_text(str(line), font, max_width)
                for wrapped_line in wrapped_lines:
                    if y_offset < rect.y + rect.height - 20:  # 再次检查高度
                        # 文字阴影
                        text_shadow = font.render(wrapped_line, True, (0, 0, 0, 80))
                        surface.blit(text_shadow, (rect.x + 13, y_offset + 1))

                        # 主文字
                        text_surface = font.render(wrapped_line, True, self.colors['text_secondary'])
                        surface.blit(text_surface, (rect.x + 12, y_offset))
                        y_offset += 18
                    else:
                        break  # 超出面板高度，停止绘制

    def wrap_text(self, text, font, max_width):
        """文本自动折行处理"""
        if not text:
            return ['']

        # 检查整行文本是否超出宽度
        if font.size(text)[0] <= max_width:
            return [text]

        # 如果超出，尝试按空格拆分
        words = text.split(' ')
        lines = []
        current_line = ''

        for word in words:
            test_line = current_line + (' ' if current_line else '') + word
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # 单个单词就已经超出，强制截断
                    lines.append(self.truncate_text(word, font, max_width))
                    current_line = ''

        if current_line:
            lines.append(current_line)

        return lines if lines else ['']

    def truncate_text(self, text, font, max_width):
        """强制截断过长的文本"""
        if font.size(text)[0] <= max_width:
            return text

        # 逐字符截断，直到符合宽度要求
        for i in range(len(text), 0, -1):
            truncated = text[:i - 3] + '...'  # 留出省略号空间
            if font.size(truncated)[0] <= max_width:
                return truncated

        return '...'  # 最极端情况

    def draw_timer_display(self, surface, rect, time_left, player_name, is_active=False):
        """绘制豪华计时器显示"""
        # 选择颜色
        if player_name == "红方":
            bg_color = self.colors['red_player']
            accent_color = self.colors['accent_tertiary']
        else:
            bg_color = self.colors['black_player']
            accent_color = self.colors['accent_secondary']

        if is_active:
            bg_color = tuple(min(255, c + 40) for c in bg_color)

        # 绘制深度阴影
        for i in range(3, 0, -1):
            shadow_rect = pygame.Rect(rect.x + i, rect.y + i, rect.width, rect.height)
            alpha = 30 - i * 8
            self.draw_rounded_rect_with_alpha(surface, shadow_rect, (0, 0, 0, alpha), 10)

        # 绘制发光效果（激活状态）
        if is_active:
            glow_rect = pygame.Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2)
            self.draw_rounded_rect_with_alpha(surface, glow_rect, (*accent_color[:3], 80), 11)

        # 绘制计时器背景渐变
        timer_surface = pygame.Surface((rect.width, rect.height))
        timer_rect = pygame.Rect(0, 0, rect.width, rect.height)
        darker_bg = tuple(max(0, c - 30) for c in bg_color)
        self.draw_premium_gradient_rect(timer_surface, timer_rect, bg_color, darker_bg, bg_color)
        surface.blit(timer_surface, (rect.x, rect.y))

        # 绘制高光
        highlight_rect = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height // 3)
        self.draw_rounded_rect_with_alpha(surface, highlight_rect, self.colors['glass'], 8)

        # 绘制边框
        border_color = accent_color if is_active else self.colors['border']
        self.draw_rounded_rect(surface, rect, (0, 0, 0, 0), 10, border_color, 2)

        # 绘制时间文本
        font = pygame.font.Font(self.config.resource.font_path, 16)
        time_text = f"{player_name}: {self.format_time(time_left)}"

        # 文字阴影
        shadow_surface = font.render(time_text, True, (0, 0, 0, 150))
        shadow_rect = shadow_surface.get_rect(center=(rect.centerx + 1, rect.centery + 1))
        surface.blit(shadow_surface, shadow_rect)

        # 主文字
        text_surface = font.render(time_text, True, self.colors['text_primary'])
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

        # 时间警告效果
        if time_left < 300:  # 5分钟以下
            warning_color = self.colors['warning'] if time_left > 60 else self.colors['error']
            # 绘制警告边框
            for i in range(3):
                warning_rect = pygame.Rect(rect.x - i, rect.y - i, rect.width + 2 * i, rect.height + 2 * i)
                self.draw_rounded_rect(surface, warning_rect, (0, 0, 0, 0), 10 + i, warning_color, 1)

            # 添加脉冲效果（通过透明度）
            pulse_alpha = int(50 + 30 * abs(time.time() % 2 - 1))
            pulse_rect = pygame.Rect(rect.x - 2, rect.y - 2, rect.width + 4, rect.height + 4)
            self.draw_rounded_rect_with_alpha(surface, pulse_rect, (*warning_color[:3], pulse_alpha), 12)

    def draw_evaluation_panel(self, surface, rect):
        """绘制评估面板"""
        eval_lines = []
        if hasattr(self, 'nn_value') and self.nn_value:
            # 根据评估值显示不同颜色的符号
            if self.nn_value > 0.3:
                eval_lines.append(f"🔴 评估: +{self.nn_value:.3f}")
            elif self.nn_value < -0.3:
                eval_lines.append(f"⚫ 评估: {self.nn_value:.3f}")
            else:
                eval_lines.append(f"🟡 评估: {self.nn_value:.3f}")

        if hasattr(self, 'mcts_moves') and self.mcts_moves:
            eval_lines.append("🔍 最佳走法:")
            sorted_moves = sorted(self.mcts_moves.items(), key=lambda x: x[1][0], reverse=True)
            for i, (move, data) in enumerate(sorted_moves[:2]):  # 只显示前2个
                visit_count, q_value, prior = data
                confidence = "高" if visit_count > 100 else "中" if visit_count > 50 else "低"
                eval_lines.append(f"  {i + 1}. {move} ({confidence})")

        if hasattr(self.config, 'play') and hasattr(self.config.play, 'simulation_num_per_move'):
            eval_lines.append(f"🏃 模拟: {self.config.play.simulation_num_per_move}次")

        if not eval_lines:
            eval_lines = ["⏳ 等待AI分析..."]

        self.draw_info_panel(surface, rect, "📊 AI 评估", eval_lines)

    def draw_operation_feedback_panel(self, widget_background):
        """绘制操作反馈面板"""
        # 只在有反馈信息且时间不超过5秒时显示
        if self.operation_feedback['message'] and (time.time() - self.operation_feedback['timestamp'] < 5):
            feedback_rect = pygame.Rect(20, self.screen_height - self.height - 30, 500, 25)

            # 根据反馈类型选择颜色
            color_map = {
                'success': self.colors['success'],
                'warning': self.colors['warning'],
                'error': self.colors['error'],
                'info': self.colors['info']
            }
            bg_color = color_map.get(self.operation_feedback['type'], self.colors['info'])

            # 绘制半透明背景
            s = pygame.Surface((feedback_rect.width, feedback_rect.height))
            s.set_alpha(220)
            s.fill(bg_color)
            widget_background.blit(s, (feedback_rect.x, feedback_rect.y))

            # 绘制边框
            pygame.draw.rect(widget_background, bg_color, feedback_rect, 2, 5)

            # 绘制文字 - 限制文本长度
            font = pygame.font.Font(self.config.resource.font_path, 14)
            message = self.operation_feedback['message']

            # 如果文本太长，进行截断
            max_width = feedback_rect.width - 20
            if font.size(message)[0] > max_width:
                message = self.truncate_text(message, font, max_width)

            text_surface = font.render(message, True, self.colors['text_primary'])
            text_rect = text_surface.get_rect(center=feedback_rect.center)
            widget_background.blit(text_surface, text_rect)

    def choose_first_player(self):
        """显示选择先手玩家的界面"""
        pygame.init()

        screen_width = 600
        screen_height = 400
        screen = pygame.display.set_mode([screen_width, screen_height], 0, 32)
        pygame.display.set_caption("选择先手玩家 - 智能象棋AI")

        try:
            icon = load_image('RK.gif', 'Piece')
            pygame.display.set_icon(icon)
        except:
            pass

        background = pygame.Surface([screen_width, screen_height])
        background.fill((240, 240, 240))

        button_width = 200
        button_height = 80
        button_spacing = 40

        total_width = button_width * 2 + button_spacing
        start_x = (screen_width - total_width) // 2
        button_y = screen_height // 2

        human_first_rect = pygame.Rect(start_x, button_y, button_width, button_height)
        ai_first_rect = pygame.Rect(start_x + button_width + button_spacing, button_y, button_width, button_height)

        # 标题位置
        title_y = screen_height // 2 - 120

        clock = pygame.time.Clock()
        choice_made = False
        human_move_first = True  # 默认人类先手

        while not choice_made:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    if human_first_rect.collidepoint(mouse_x, mouse_y):
                        human_move_first = True
                        choice_made = True
                        logger.info("🎯 玩家选择：人类先手")

                    elif ai_first_rect.collidepoint(mouse_x, mouse_y):
                        human_move_first = False
                        choice_made = True
                        logger.info("🎯 玩家选择：AI先手")

            # 重绘背景
            screen.blit(background, (0, 0))

            # 绘制标题 - 简化版本
            try:
                title_font = pygame.font.Font(self.config.resource.font_path, 28)
                subtitle_font = pygame.font.Font(self.config.resource.font_path, 18)
            except:
                title_font = pygame.font.Font(None, 36)
                subtitle_font = pygame.font.Font(None, 24)

            # 主标题
            title_text = "选择先手玩家"
            title_surface = title_font.render(title_text, True, (50, 50, 50))  # 深灰色
            title_rect = title_surface.get_rect(center=(screen_width // 2, title_y))
            screen.blit(title_surface, title_rect)

            # 副标题
            subtitle_text = "请选择谁先开始下棋"
            subtitle_surface = subtitle_font.render(subtitle_text, True, (100, 100, 100))  # 中灰色
            subtitle_rect = subtitle_surface.get_rect(center=(screen_width // 2, title_y + 40))
            screen.blit(subtitle_surface, subtitle_rect)

            # 获取鼠标位置用于悬停效果
            mouse_pos = pygame.mouse.get_pos()
            human_hover = human_first_rect.collidepoint(mouse_pos)
            ai_hover = ai_first_rect.collidepoint(mouse_pos)

            # 绘制人类先手按钮
            self.draw_simple_button(screen, human_first_rect, "人类先手",
                                    "红方先行", (220, 80, 80), human_hover)

            # 绘制AI先手按钮
            self.draw_simple_button(screen, ai_first_rect, "AI先手",
                                    "让AI开局", (80, 120, 220), ai_hover)

            # 绘制说明文字 - 简化版本
            info_y = button_y + button_height + 60
            try:
                info_font = pygame.font.Font(self.config.resource.font_path, 14)
            except:
                info_font = pygame.font.Font(None, 18)

            info_lines = [
                "提示：",
                "人类先手：您执红子先行，享有开局优势",
                "AI先手：挑战更高难度，观察AI的开局策略"
            ]

            for i, line in enumerate(info_lines):
                if i == 0:
                    color = (80, 80, 80)  # 深灰色标题
                else:
                    color = (120, 120, 120)  # 中灰色文字

                info_surface = info_font.render(line, True, color)
                info_rect = info_surface.get_rect(center=(screen_width // 2, info_y + i * 25))
                screen.blit(info_surface, info_rect)

            pygame.display.flip()
            clock.tick(60)

        # 显示选择结果
        self.show_choice_confirmation(screen, human_move_first)

        return human_move_first

    def draw_simple_button(self, surface, rect, main_text, sub_text, color, hover=False):
        """绘制简单按钮"""
        # 选择颜色
        if hover:
            bg_color = tuple(min(255, c + 30) for c in color)
            border_color = (50, 50, 50)
        else:
            bg_color = color
            border_color = (100, 100, 100)

        # 绘制按钮背景
        pygame.draw.rect(surface, bg_color, rect, border_radius=10)

        # 绘制边框
        pygame.draw.rect(surface, border_color, rect, 2, border_radius=10)

        # 绘制文字
        try:
            main_font = pygame.font.Font(self.config.resource.font_path, 20)
            sub_font = pygame.font.Font(self.config.resource.font_path, 14)
        except:
            main_font = pygame.font.Font(None, 24)
            sub_font = pygame.font.Font(None, 18)

        # 使用白色文字
        text_color = (255, 255, 255)

        # 绘制主文字
        main_surface = main_font.render(main_text, True, text_color)
        main_rect = main_surface.get_rect(center=(rect.centerx, rect.centery - 12))
        surface.blit(main_surface, main_rect)

        # 绘制副文字
        sub_surface = sub_font.render(sub_text, True, text_color)
        sub_rect = sub_surface.get_rect(center=(rect.centerx, rect.centery + 15))
        surface.blit(sub_surface, sub_rect)

    def show_choice_confirmation(self, screen, human_move_first):
        """显示选择确认 - 简化版本"""
        # 创建确认信息
        try:
            font = pygame.font.Font(self.config.resource.font_path, 24)
            loading_font = pygame.font.Font(self.config.resource.font_path, 16)
        except:
            font = pygame.font.Font(None, 32)
            loading_font = pygame.font.Font(None, 20)

        if human_move_first:
            text = "已选择：人类先手（红方）"
            color = (220, 80, 80)  # 红色
        else:
            text = "已选择：AI先手（红方）"
            color = (80, 120, 220)  # 蓝色

        # 创建半透明覆盖层
        overlay = pygame.Surface((screen.get_width(), screen.get_height()))
        overlay.set_alpha(200)
        overlay.fill((240, 240, 240))  # 浅色半透明背景
        screen.blit(overlay, (0, 0))

        # 绘制确认文字
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        screen.blit(text_surface, text_rect)

        # 绘制加载提示
        loading_text = "正在准备游戏..."
        loading_surface = loading_font.render(loading_text, True, (100, 100, 100))
        loading_rect = loading_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 40))
        screen.blit(loading_surface, loading_rect)

        pygame.display.flip()
        sleep(1.5)  # 显示1.5秒确认信息

    def show_game_over_dialog(self, game_screen, widget_background):
        """显示游戏结束对话框"""
        dialog_width = 800
        dialog_height = 600
        dialog_screen = pygame.display.set_mode([dialog_width, dialog_height], 0, 32)
        pygame.display.set_caption("游戏结束 - 智能象棋AI")

        try:
            icon = load_image('RK.gif', 'Piece')
            pygame.display.set_icon(icon)
        except:
            pass

        # 更新游戏统计
        self.update_game_stats()

        # 获取胜利者信息
        winner_info = self.get_winner_info()

        # 获取完整棋谱
        game_record = self.get_formatted_game_record()

        clock = pygame.time.Clock()
        dialog_running = True

        # 按钮设置
        button_width = 140
        button_height = 50
        button_spacing = 30
        buttons_y = dialog_height - 100

        # 两个按钮的布局
        total_width = button_width * 2 + button_spacing
        start_x = (dialog_width - total_width) // 2

        print_record_rect = pygame.Rect(start_x, buttons_y, button_width, button_height)
        exit_rect = pygame.Rect(start_x + button_width + button_spacing, buttons_y, button_width, button_height)

        while dialog_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    if print_record_rect.collidepoint(mouse_x, mouse_y):
                        logger.info("用户选择打印棋谱")
                        self.print_and_save_record()

                    elif exit_rect.collidepoint(mouse_x, mouse_y):
                        logger.info("用户选择退出游戏")
                        pygame.quit()
                        sys.exit()

            # 绘制背景
            dialog_screen.fill(self.colors['bg_primary'])

            # 绘制胜利信息
            self.draw_winner_info(dialog_screen, winner_info)

            # 绘制游戏统计
            self.draw_final_stats(dialog_screen)

            # 绘制棋谱
            self.draw_complete_record(dialog_screen, game_record)

            # 绘制按钮
            mouse_pos = pygame.mouse.get_pos()
            print_record_hover = print_record_rect.collidepoint(mouse_pos)
            exit_hover = exit_rect.collidepoint(mouse_pos)

            self.draw_dialog_button(dialog_screen, print_record_rect, "打印棋谱", (59, 130, 246), print_record_hover)
            self.draw_dialog_button(dialog_screen, exit_rect, "退出游戏", (239, 68, 68), exit_hover)

            pygame.display.flip()
            clock.tick(60)

    def update_game_stats(self):
        """更新游戏统计数据"""
        self.game_stats['total_games'] += 1

        if self.env.board.winner:
            if self.env.board.winner.name == 'red':
                if self.human_move_first:
                    self.game_stats['player_wins'] += 1
                else:
                    self.game_stats['ai_wins'] += 1
            elif self.env.board.winner.name == 'black':
                if self.human_move_first:
                    self.game_stats['ai_wins'] += 1
                else:
                    self.game_stats['player_wins'] += 1
            else:
                self.game_stats['draws'] += 1

        # 计算总移动数
        if hasattr(self.env.board, 'record') and self.env.board.record:
            self.game_stats['total_moves'] += len([m for m in self.env.board.record.split('\n') if m.strip()])

    def is_winner_definitive(self):
        """检查胜负是否真正确定"""
        if not self.env.board.winner:
            return False
        
        # 检查是否是有效的胜负结果
        winner_name = self.env.board.winner.name if hasattr(self.env.board.winner, 'name') else None
        
        # 只有明确的红方或黑方胜利才算确定
        if winner_name in ['red', 'black']:
            # 进一步检查棋盘状态是否支持这个结果
            red_king = self.env.board.get_chessman_by_name('red_king')
            black_king = self.env.board.get_chessman_by_name('black_king')
            
            # 如果一方的王被吃掉，胜负确定
            if not red_king or not black_king:
                return True
            
            # 如果是因为时间到而判定的胜负，也算确定
            if hasattr(self, 'red_time_left') and hasattr(self, 'black_time_left'):
                if self.red_time_left <= 0 or self.black_time_left <= 0:
                    return True
            
            # 其他情况下的红黑胜利也算确定
            return True
        
        # draw类型的胜负需要特殊检查
        elif winner_name == 'draw':
            return True  # 明确的和棋也算确定
        
        # 其他情况都算不确定
        logger.warning(f"胜负状态不确定: winner={self.env.board.winner}, name={winner_name}")
        return False

    def continue_game_after_uncertainty(self, screen, widget_background):
        """在胜负不确定的情况下继续游戏"""
        logger.info("🔄 重新启动游戏循环...")
        
        # 重置AI状态
        if hasattr(self.ai, 'search_results'):
            self.ai.search_results = {}
        
        # 显示提示信息给用户
        framerate = pygame.time.Clock()
        for _ in range(60):  # 显示1秒的提示信息
            self.draw_widget(screen, widget_background)
            framerate.tick(60)
            self.chessmans.clear(screen, widget_background)
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()
        
        # 重新进入主游戏循环
        self.start_main_game_loop(screen, widget_background)

    def start_main_game_loop(self, screen, widget_background):
        """启动主游戏循环（提取出来便于重用）"""
        framerate = pygame.time.Clock()
        current_chessman = None
        board_background = pygame.Surface(screen.get_size())
        
        while not self.env.board.is_end():
            # 时间控制逻辑
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

            # 完整的事件处理逻辑
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.board.print_record()
                    self.ai.close(wait=False)
                    result = self.env.board.get_result_string()
                    red_team = "玩家"
                    black_team = "AI"
                    self.env.board.save_record("", "", red_team, black_team, result)
                    logger.info(f"游戏被用户关闭，棋谱已保存")
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.locals.VIDEORESIZE:
                    pass
                elif event.type == pygame.locals.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if self.hittest(mouse_x, mouse_y, (20, self.height + 20, 120, 45)):
                        if self.is_human_turn():
                            logger.info("🔄 玩家请求悔棋")
                            print("悔棋!\n")
                            self.set_operation_feedback("🔄 正在执行悔棋...", 'info')
                            # 悔棋功能需要复杂的状态管理，在此简化处理
                            logger.warning("⚠️ 在恢复模式下暂不支持悔棋")
                            # 悔棋后清除当前选中的棋子
                            current_chessman = None
                            self.set_operation_feedback("⚠️ 悔棋功能暂不可用", 'warning')
                        else:
                            logger.info("⚠️ AI回合时无法悔棋")
                            print("AI思考中，无法悔棋！")
                            self.set_operation_feedback("⚠️ AI回合无法悔棋！", 'warning')
                    elif self.hittest(mouse_x, mouse_y, (160, self.height + 20, 150, 45)):
                        logger.info("📄 玩家请求保存棋谱")
                        print("保存棋谱!")
                        self.set_operation_feedback("📄 正在保存棋谱...", 'info')
                        self.env.board.print_record()
                        try:
                            result = self.env.board.get_result_string()
                            red_team = "玩家"
                            black_team = "AI"
                            self.env.board.save_record("", "", red_team, black_team, result)
                            logger.info(f"✅ 棋谱已保存")
                            print(f"棋谱已保存")
                            self.set_operation_feedback(f"✅ 棋谱已保存！", 'success')
                        except Exception as e:
                            logger.error(f"❌ 棋谱保存失败: {str(e)}")
                            print(f"棋谱保存失败: {str(e)}")
                            self.set_operation_feedback("❌ 棋谱保存失败！", 'error')
                    elif self.is_human_turn():  # 使用新的判断函数
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
                                        if success:
                                            self.chessmans.remove(chessman_sprite)
                                            chessman_sprite.kill()
                                            current_chessman.is_selected = False
                                            current_chessman = None
                                            self.history.append(self.env.get_state())
                                            # 检查三次重复局面
                                            if self.check_threefold_repetition():
                                                break  # 如果判定和棋，跳出循环
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
                                    if success:
                                        current_chessman.is_selected = False
                                        current_chessman = None
                                        self.history.append(self.env.get_state())
                                        # 切换到黑方计时
                                        self.current_timer = 'black'
                                        self.last_move_time = time.time()
            
            self.draw_widget(screen, widget_background)
            framerate.tick(20)
            self.chessmans.clear(screen, board_background)
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()

    def get_winner_info(self):
        """获取胜利者信息"""
        if not self.env.board.winner:
            return {"title": "游戏结束", "subtitle": "平局", "color": self.colors['text_secondary']}

        winner_name = self.env.board.winner.name
        if winner_name == 'red':
            if self.human_move_first:
                return {"title": "恭喜获胜！", "subtitle": "红方胜利 - 人类玩家获胜", "color": self.colors['success']}
            else:
                return {"title": "AI获胜", "subtitle": "红方胜利 - AI获胜", "color": self.colors['error']}
        elif winner_name == 'black':
            if self.human_move_first:
                return {"title": "AI获胜", "subtitle": "黑方胜利 - AI获胜", "color": self.colors['error']}
            else:
                return {"title": "恭喜获胜！", "subtitle": "黑方胜利 - 人类玩家获胜", "color": self.colors['success']}
        else:
            return {"title": "游戏结束", "subtitle": "平局", "color": self.colors['text_secondary']}

    def get_formatted_game_record(self):
        """获取格式化的完整棋谱"""
        if not hasattr(self.env.board, 'record') or not self.env.board.record:
            return ["无棋谱记录"]

        moves = self.env.board.record.strip().split('\n')
        formatted_moves = []

        for i, move in enumerate(moves, 1):
            if move.strip():
                formatted_moves.append(f"{i:2d}. {move.strip()}")

        return formatted_moves if formatted_moves else ["无棋谱记录"]

    def draw_winner_info(self, screen, winner_info):
        """绘制胜利者信息"""
        title_y = 50

        try:
            title_font = pygame.font.Font(self.config.resource.font_path, 36)
            subtitle_font = pygame.font.Font(self.config.resource.font_path, 24)
        except:
            title_font = pygame.font.Font(None, 44)
            subtitle_font = pygame.font.Font(None, 32)

        # 绘制主标题
        title_surface = title_font.render(winner_info["title"], True, winner_info["color"])
        title_rect = title_surface.get_rect(center=(screen.get_width() // 2, title_y))
        screen.blit(title_surface, title_rect)

        # 绘制副标题
        subtitle_surface = subtitle_font.render(winner_info["subtitle"], True, self.colors['text_secondary'])
        subtitle_rect = subtitle_surface.get_rect(center=(screen.get_width() // 2, title_y + 50))
        screen.blit(subtitle_surface, subtitle_rect)

    def draw_final_stats(self, screen):
        """绘制最终统计信息"""
        stats_y = 130

        try:
            stats_font = pygame.font.Font(self.config.resource.font_path, 18)
        except:
            stats_font = pygame.font.Font(None, 24)

        total_moves = len([m for m in self.env.board.record.split('\n') if m.strip()]) if hasattr(self.env.board,
                                                                                                  'record') and self.env.board.record else 0

        # 使用游戏结束时间来计算总用时，避免时间一直增加
        if self.game_stats.get('current_game_end') and self.game_stats.get('current_game_start'):
            game_time = self.game_stats['current_game_end'] - self.game_stats['current_game_start']
        else:
            game_time = 0

        stats_lines = [
            f"本局步数: {total_moves}",
            f"用时: {int(game_time // 60):02d}:{int(game_time % 60):02d}",
            f"历史战绩: {self.game_stats['player_wins']}胜 {self.game_stats['ai_wins']}败 {self.game_stats['draws']}平"
        ]

        for i, line in enumerate(stats_lines):
            stats_surface = stats_font.render(line, True, self.colors['text_primary'])
            stats_rect = stats_surface.get_rect(center=(screen.get_width() // 2, stats_y + i * 30))
            screen.blit(stats_surface, stats_rect)

    def draw_complete_record(self, screen, moves):
        """绘制完整棋谱"""
        record_start_y = 250
        record_height = 280

        # 绘制棋谱标题
        try:
            title_font = pygame.font.Font(self.config.resource.font_path, 20)
            move_font = pygame.font.Font(self.config.resource.font_path, 14)
        except:
            title_font = pygame.font.Font(None, 26)
            move_font = pygame.font.Font(None, 18)

        title_surface = title_font.render("完整棋谱", True, self.colors['accent'])
        title_rect = title_surface.get_rect(center=(screen.get_width() // 2, record_start_y))
        screen.blit(title_surface, title_rect)

        # 绘制棋谱背景
        record_rect = pygame.Rect(50, record_start_y + 30, screen.get_width() - 100, record_height)
        pygame.draw.rect(screen, self.colors['bg_secondary'], record_rect, border_radius=10)
        pygame.draw.rect(screen, self.colors['border'], record_rect, 2, border_radius=10)

        # 分列显示棋谱
        cols = 3
        col_width = (record_rect.width - 40) // cols
        moves_per_col = min(25, len(moves) // cols + 1)

        for col in range(cols):
            start_idx = col * moves_per_col
            end_idx = min(start_idx + moves_per_col, len(moves))
            col_moves = moves[start_idx:end_idx]

            x = record_rect.x + 20 + col * col_width
            y = record_start_y + 50

            for i, move in enumerate(col_moves):
                if y < record_start_y + record_height - 20:
                    move_surface = move_font.render(move, True, self.colors['text_primary'])
                    screen.blit(move_surface, (x, y))
                    y += 20

    def draw_dialog_button(self, screen, rect, text, color, hover=False):
        """绘制对话框按钮"""
        if hover:
            bg_color = tuple(min(255, c + 30) for c in color)
        else:
            bg_color = color

        # 绘制按钮
        pygame.draw.rect(screen, bg_color, rect, border_radius=10)
        pygame.draw.rect(screen, self.colors['border_light'], rect, 2, border_radius=10)

        # 绘制文字
        try:
            font = pygame.font.Font(self.config.resource.font_path, 18)
        except:
            font = pygame.font.Font(None, 24)

        text_surface = font.render(text, True, self.colors['text_primary'])
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)

    def print_and_save_record(self):
        """打印并保存棋谱"""
        try:
            # 打印到控制台
            print("\n" + "=" * 50)
            print("📋 完整棋谱")
            print("=" * 50)
            self.env.board.print_record()
            print("=" * 50)

            result = self.env.board.get_result_string()
            red_team = "玩家"
            black_team = "AI"
            self.env.board.save_record("", "", red_team, black_team, result)

            # 计算游戏信息
            total_moves = len([m for m in self.env.board.record.split('\n') if m.strip()]) if hasattr(self.env.board,
                                                                                                      'record') and self.env.board.record else 0
            if self.game_stats.get('current_game_end') and self.game_stats.get('current_game_start'):
                game_time = self.game_stats['current_game_end'] - self.game_stats['current_game_start']
            else:
                game_time = 0

            winner_text = "平局"
            if self.env.board.winner:
                if self.env.board.winner.name == 'red':
                    winner_text = "红方胜利"
                elif self.env.board.winner.name == 'black':
                    winner_text = "黑方胜利"

            print(f"🏆 胜负结果: {winner_text}")
            print(f"📊 总步数: {total_moves}")
            print(f"⏱️  用时: {int(game_time // 60):02d}:{int(game_time % 60):02d}")
            print(f"💾 棋谱已保存")
            print("=" * 50 + "\n")

            logger.info(f"✅ 棋谱已重新打印并保存")

        except Exception as e:
            print(f"❌ 打印棋谱时出错: {str(e)}")
            logger.error(f"❌ 打印棋谱失败: {str(e)}")


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
                images = load_images("RR.gif", "RRS.gif")
            elif isinstance(chess, Cannon):
                images = load_images("RC.gif", "RCS.gif")
            elif isinstance(chess, Knight):
                images = load_images("RN.gif", "RNS.gif")
            elif isinstance(chess, King):
                images = load_images("RK.gif", "RKS.gif")
            elif isinstance(chess, Elephant):
                images = load_images("RB.gif", "RBS.gif")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.gif", "RAS.gif")
            else:
                images = load_images("RP.gif", "RPS.gif")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.gif", "BRS.gif")
            elif isinstance(chess, Cannon):
                images = load_images("BC.gif", "BCS.gif")
            elif isinstance(chess, Knight):
                images = load_images("BN.gif", "BNS.gif")
            elif isinstance(chess, King):
                images = load_images("BK.gif", "BKS.gif")
            elif isinstance(chess, Elephant):
                images = load_images("BB.gif", "BBS.gif")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.gif", "BAS.gif")
            else:
                images = load_images("BP.gif", "BPS.gif")
        chessman_sprite = Chessman_Sprite(images, chess, w, h)
        sprite_group.add(chessman_sprite)


def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None


def translate_hit_area(screen_x, screen_y, w=80, h=80):
    return screen_x // w, 9 - screen_y // h
