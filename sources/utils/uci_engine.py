import os
import gc
import subprocess
import sys
import time
import threading
import shlex
import queue
import re
from sources.BookHandler.BookHandler import BookHandler

import requests


class Engine_Manager:
    # def __int__(self, engine_path, HashSize = 4096 , Threads = 16):
    #     self.engine_path = engine_path
    #     self.HashSize = HashSize
    #     self.Threads = Threads
    #     self.ready = False
    #     self.engine = subprocess.Popen(engine_path,
    #                          cwd=os.path.dirname(engine_path),
    #                          stdin=subprocess.PIPE,
    #                          stdout=subprocess.PIPE,
    #                          stderr=subprocess.PIPE,
    #                          universal_newlines=True)
    #     out = self.send("uci")
    #     if out[-1] == "uciok":
    #         self.ready = True
    #     else:
    #         self.engine.kill()
    #     self.set_option("Hash", self.HashSize)
    #     self.set_option("Threads", self.Threads)
    # def send(self, Command):
    #     return self.engine.communicate(Command + '\n')
    #
    # def set_option(self, key, value):
    #     self.send(f"setoption name {key} value {value}")
    #
    # def get_move(self, fen, IsRedGo, time = 5000 , depth = 50):
    #     self.send(f'position fen {fen}\n')
    #     self.send("go movetime " + str(time) + "\n")

    @staticmethod
    def get_uci_move(engine_path, moves, threads, IsRedGo, time):
        p = subprocess.Popen(engine_path,
                             cwd=os.path.dirname(engine_path),
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
        p.stdin.write('uci\n')
        p.stdin.write("setoption name Threads value " + str(threads) + "\n")

        fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w moves '
        for move in moves:
            fen += move + ' '
        p.stdin.write(f'position fen {fen}\n')
        # print(f'position fen {fen}\n')
        p.stdin.write("go movetime " + str(time * 1000) + "\n")
        p.stdin.flush()
        count = 0
        while True:
            text = p.stdout.readline().strip()
            if text != "":
                try:
                    if text.startswith("bestmove"):
                        # print(text)
                        text = text.split(' ')[1]
                        # if not IsRedGo:
                        #     text = BookHandler.MirrorMoveLeftRight(text)
                        p.kill()
                        return text
                except Exception as e:
                    print(e)
                    p.kill()
                    return None
            else:
                count += 1
                if count > 1:
                    p.kill()
                    return None

# print(Engine_Manager.get_uci_move("D:\Document\Code\my project\Chess\ChessAI\data\Engine\pikafish-avx512.exe",
#                                 "2rakabr1/3n5/1c2b3n/p1p5p/3Np1p2/2P6/P5P1P/4B1C2/4AN3/1RBAKR3", 32, 1, 5))

class EngineHelper:
    def __init__(self, engine_path, configs=None):
        self.engine_type = "uci"
        self.engine_path = engine_path
        self.engine_name = ""
        self.engine_author = ""
        self.engine = None
        self.last_best_move = ""
        self.last_ponder_move = ""
        self.ban_moves = ""
        self.best_move_event = None
        self.info_event = None
        self.option_list = []
        self.thread_handle_output = None
        self.configs = {} if configs is None else configs
        self.analyze_queue = queue.Queue()
        self.skip_list = []
        self.last_output_time = 0
        self.ignore_move = ""
        self.initialized = False

    def stop(self):
        if self.engine is not None:
            self.engine.terminate()

    def init(self):
        self.initialized = False
        print("Init Engine")
        self.engine = subprocess.Popen(
            self.engine_path,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )
        self.engine.stdin.write("uci\n")
        self.engine.stdin.write("ucci\n")
        self.option_list.clear()
        self.send_command("uci")
        self.send_command("ucci")
        self.thread_handle_output = threading.Thread(target=self.wait_for_exit)
        self.thread_handle_output.start()
        self.engine.stdout.readline()
        self.initialized = True

    def wait_for_exit(self):
        self.engine.wait()

    def handle_output_line(self, line):
        if not line:
            return
        parts = line.split(" ")
        cmd = parts[0]
        if cmd == "info":
            infos = {}
            for i in range(1, len(parts)):
                if parts[i] == "pv":
                    infos[parts[i]] = " ".join(parts[i + 1:])
                    break
                elif parts[i] == "string":
                    infos[parts[i]] = " ".join(parts[i + 1:])
                    break
                elif parts[i] == "score":
                    if parts[i + 1] == "cp":
                        infos[parts[i]] = parts[i + 2]
                        i += 2
                    elif parts[i + 1] == "mate":
                        infos[parts[i]] = f"绝杀 ({parts[i + 2]})"
                        i += 2
                    else:
                        infos[parts[i]] = parts[i + 1]
                        i += 1
                else:
                    if len(parts) > i + 1 and parts[i + 1] not in infos:
                        infos[parts[i]] = parts[i + 1]
                        i += 1
                    else:
                        infos[parts[i]] = ""
            self.info_event(cmd, infos)
        elif cmd == "bestmove":
            source_fen = ""
            if not self.analyze_queue.empty():
                source_fen = self.analyze_queue.get()
            if source_fen in self.skip_list:
                self.skip_list.remove(source_fen)
                return
            if len(parts) > 2:
                self.best_move_event(source_fen, parts[1], parts[3])
            else:
                self.best_move_event(source_fen, parts[1], "")
        elif cmd == "option":
            self.option_list.append(line)
        elif cmd == "id":
            if len(parts) >= 3:
                type_ = parts[1].lower()
                if type_ == "name":
                    self.engine_name = " ".join(parts[2:])
                elif type_ == "author":
                    self.engine_author = " ".join(parts[2:])
        elif cmd == "ucciok":
            self.engine_type = "ucci"

    def set_option(self, key, value):
        if self.engine_type == "ucci":
            self.send_command(f"setoption {key} {value}")
        else:
            self.send_command(f"setoption name {key} value {value}")

    def stop_analyze(self):
        self.send_command("stop")

    def stop_analyze_and_skip(self, skip_fen):
        self.send_command("stop")
        self.skip_list.add(skip_fen)

    def ponder_miss(self, skip_fen):
        self.send_command("stop")
        self.skip_list.add(skip_fen)

    def ponder_hit(self):
        print("Ponder hit")
        self.send_command("ponderhit")

    def send_command(self, cmd):
        print(f"Engine Input: {cmd}")
        self.engine.stdin.write(f"{cmd}\n")

    def start_analyze_ponder(self, fen, time_sec, depth):
        if not self.initialized:
            self.init()
        self.analyze_queue.put(fen)
        print(f"Start Analyzing Ponder: \n{fen}")
        self.send_command(f"position fen {fen}")
        if self.ban_moves:
            self.send_command(f"banmoves {self.ban_moves}")
            self.ban_moves = ""
        if time_sec > 0:
            self.send_command(f"go ponder movetime {int(time_sec * 1000)} depth {depth}")
        else:
            self.send_command(f"go ponder depth {depth}")

    def start_analyze(self, fen, time_sec, depth):
        if not self.initialized or not self.engine or not self.engine.handle or self.engine.has_exited:
            self.init()
        self.analyze_queue.put(fen)
        print(f"Start Analyzing: \n{fen}")
        self.send_command(f"position fen {fen}")
        if self.ban_moves:
            self.send_command(f"banmoves {self.ban_moves}")
            self.ban_moves = ""
        if time_sec > 0:
            self.send_command(f"go movetime {int(time_sec * 1000)} depth {depth}")
        else:
            self.send_command(f"go depth {depth}")

