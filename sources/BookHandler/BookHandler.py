import logging
import struct
import os
from functools import cmp_to_key

from sources.BookHandler.BookUtils import BookUtils
import sqlite3

class BookData:
    def __init__(self, move, word, score, win_rate, win_num, draw_num, lose_num, note, source):
        self.move = move
        self.word = word
        self.score = score
        self.win_rate = win_rate
        self.win_num = win_num
        self.draw_num = draw_num
        self.lose_num = lose_num
        self.note = note
        self.source = source

    def __hash__(self):
        return hash(self.move)

    def __eq__(self, other):
        if self.move == other.move and self.score == other.score:
            return True
        else:
            return False

    def get_source(self):
        return self.source

    def set_source(self, source):
        self.source = source

    def __str__(self):
        return f"BookData{{move='{self.move}', score={self.score}, win_rate={self.win_rate}, win_num={self.win_num}, draw_num={self.draw_num}, lose_num={self.lose_num}, note='{self.note}', source='{self.source}'}}"

    def get_word(self):
        return self.word

    def set_word(self, word):
        self.word = word

    def get_move(self):
        return self.move

    def set_move(self, move):
        self.move = move

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def get_win_rate(self):
        return self.win_rate

    def set_win_rate(self, win_rate):
        self.win_rate = win_rate

    def get_win_num(self):
        return self.win_num

    def set_win_num(self, win_num):
        self.win_num = win_num

    def get_draw_num(self):
        return self.draw_num

    def set_draw_num(self, draw_num):
        self.draw_num = draw_num

    def get_lose_num(self):
        return self.lose_num

    def set_lose_num(self, lose_num):
        self.lose_num = lose_num

    def get_note(self):
        return self.note

    def set_note(self, note):
        self.note = note


class BookHandler:
    OBUtils = BookUtils()


    @staticmethod
    def MirrorFenRedBlack(fen):
        # rnbaka2r/9/1c2b1nc1/p1p1p1p1p/9/2P6/P3P1P1P/1CN4C1/9/R1BAKABNR w - - 0 1
        args = fen.split(' ')
        board = args[0]
        rows = board.split('/')
        new_rows = []
        for i in range(len(rows) - 1, -1, -1):
            new_row = ""
            for c in rows[i]:
                s = c
                if 'a' <= c <= 'z':
                    new_row += s.upper()
                else:
                    new_row += s.lower()
            new_rows.append(new_row)
        new_board = '/'.join(new_rows)
        return new_board

    @staticmethod
    def MirrorFenLeftRight(fen):
        # rnbaka2r/9/1c2b1nc1/p1p1p1p1p/9/2P6/P3P1P1P/1CN4C1/9/R1BAKABNR w - - 0 1
        args = fen.split(' ')
        board = args[0]
        rows = board.split('/')
        new_rows = []
        for row in rows:
            new_row = row[::-1]
            new_rows.append(new_row)
        new_board = '/'.join(new_rows)
        return new_board + ' ' + ' '.join(args[1:])

    @staticmethod
    def MirrorMoveLeftRight(move):
        return chr(ord('i') - ord(move[0]) + ord('a')) + move[1] + chr(ord('i') - ord(move[2]) + ord('a')) + move[3]

    @staticmethod
    def Move2Point(move):
        # h2e2
        x0 = ord(move[0]) - ord('a')
        y0 = ord(move[1]) - ord('0')
        x1 = ord(move[2]) - ord('a')
        y1 = ord(move[3]) - ord('0')
        return x0, y0, x1, y1

    @staticmethod
    def Point2Move(from_point, to_point):
        letters = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(' ')
        move = ""
        move += letters[from_point[0]]
        move += str(9 - from_point[1])
        move += letters[to_point[0]]
        move += str(9 - to_point[1])
        return move

    @staticmethod
    def QueryAll(open_book_list, fen):
        result_list = []
        for openbook in open_book_list.values():
            results = openbook.Query(fen)
            if results:
                result_list.extend(results)
                continue
            results = openbook.Query(BookHandler.MirrorFenRedBlack(fen))
            if results:
                for r in results:
                    from_point = BookHandler.Move2Point(r.Move[:2], True)
                    to_point = BookHandler.Move2Point(r.Move[2:], True)
                    from_point.y = 9 - from_point.y
                    to_point.y = 9 - to_point.y
                    r.Move = BookHandler.Point2Move(from_point, to_point)
                result_list.extend(results)
                continue
            results = openbook.Query(BookHandler.MirrorFenLeftRight(fen))
            if results:
                for r in results:
                    from_point = BookHandler.Move2Point(r.Move[:2], True)
                    to_point = BookHandler.Move2Point(r.Move[2:], True)
                    from_point.x = 8 - from_point.x
                    to_point.x = 8 - to_point.x
                    r.Move = BookHandler.Point2Move(from_point, to_point)
                result_list.extend(results)
                continue
            results = openbook.Query(
                BookHandler.MirrorFenLeftRight(BookHandler.MirrorFenRedBlack(fen)))
            if results:
                for r in results:
                    from_point = BookHandler.Move2Point(r.Move[:2], True)
                    to_point = BookHandler.Move2Point(r.Move[2:], True)
                    from_point.y = 9 - from_point.y
                    to_point.y = 9 - to_point.y
                    from_point.x = 8 - from_point.x
                    to_point.x = 8 - to_point.x
                    r.Move = BookHandler.Point2Move(from_point, to_point)
                result_list.extend(results)
                continue
            if result_list:
                return result_list
        return result_list

    @staticmethod
    def fixFen(fen):
        fixed_fen = ''
        covert = {'k': 'n', 'e': 'b', 'm': 'a', 's': 'k', 'K': 'N', 'E': 'B', 'M': 'A', 'S': 'K'}
        for i in range(len(fen)):
            if fen[i] in covert:
                fixed_fen += covert[fen[i]]
            else:
                fixed_fen += fen[i]
        return fixed_fen


class Book:
    def __init__(self, file: str):
        self.FileName = file
        self.BookName = os.path.basename(file)
        self.SQLite = sqlite3.connect(file, check_same_thread = False)
        self.SQLite.text_factory = lambda x: str(x, 'utf-8', 'ignore')

    def query(self, fen, IsRedGo):
        if not IsRedGo:
            fen = BookHandler.MirrorFenRedBlack(fen)
        Zobrist = BookHandler.OBUtils.GetZobristFromFen(fen, IsRedGo, 0)
        # print(Zobrist)
        result = self.get(Zobrist, 0 if IsRedGo else 1)
        # for i in result:
        #     print(str(i))
        # print()
        Zobrist = BookHandler.OBUtils.GetZobristFromFen(fen, IsRedGo, 1)
        # print(Zobrist)
        result1 = self.get(Zobrist, 1 if IsRedGo else 0)
        # for i in result1:
        #     print(str(i))
        return list(set(result + result1))

    def get(self, zobrist, leftRightSwap):
        results = []

        sql = ""
        if zobrist >> 63 != 0:
            zobrist_double = struct.unpack('d', struct.pack('Q', zobrist))[0]
            sql = f"SELECT * FROM bhobk WHERE CAST(vkey AS TEXT) = {zobrist_double} AND vvalid = 1;"
        else:
            sql = f"SELECT * FROM bhobk WHERE CAST(vkey AS INTEGER) = {zobrist} AND vvalid = 1;"
        # print(sql)
        try:
            cursor = self.SQLite.cursor()
            cursor.execute(sql)
            for row in cursor.fetchall():
                bd = BookData(0, 0, 0, 0, 0, 0, 0, 0, 0)
                bd.set_score(row[3])
                bd.set_win_num(row[4])
                bd.set_draw_num(row[5])
                bd.set_lose_num(row[6])
                win_rate = int(10000 * (bd.get_win_num() + bd.get_draw_num() / 2.0) / (
                        max(1, bd.get_win_num() + bd.get_draw_num() + bd.get_lose_num())))
                bd.set_win_rate(win_rate / 100)
                bd.set_note(row[8])
                vmove = row[2]
                bd.set_move(BookHandler.OBUtils.ConvertVmoveToCoord(vmove, leftRightSwap))

                bd.set_source(self.BookName.split('\\')[-1])
                results.append(bd)

        except Exception as e:
            print(e)

        return results

    def __del__(self):
        self.SQLite.close()

# print(BookHandler.MirrorMoveLeftRight('b9c7'))

# print('rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR')
# print(BookHandler.fixFen('rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR'))
# print('rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR')

# book = Book('D:\Document\Code\my project\Chess\开局库\三戒开局库V2023831(1.71GB).obk')
#
# # "rnbakabnr/9/4c2c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
# # "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/4C2C1/9/RNBAKABNR"
# print(BookHandler.MirrorFenRedBlack("rnbakabnr/9/4c2c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"))
#
# result = book.query(BookHandler.MirrorFenRedBlack("rnbakabr1/9/1c4nc1/p1p1p2Rp/6p2/9/P1P1P1P1P/1C2C1N2/9/RNBAKAB2"), 0)
#
#
# def cmp(bd1: BookData, bd2: BookData):
#     if bd1.score != bd2.score:
#         return 1 if bd1.score > bd2.score else -1
#     elif bd1.win_rate != bd2.win_rate:
#         return 1 if bd1.win_rate > bd2.win_rate else -1
#     elif bd1.win_num != bd2.win_num:
#         return 1 if bd1.win_num > bd2.win_num else -1
#     else:
#         return 0
#
#
# result.sort(key = cmp_to_key(cmp), reverse = 1)
#
# for i in range(len(result)):
#     print(str(result[i]))

# db = sqlite3.connect('D:\Document\Code\my project\Chess\开局库\三戒开局库V2023831(1.71GB).obk')
# db.text_factory = lambda x: str(x, 'utf-8', 'ignore')
#
# cur = db.cursor()
#
# cur.execute("SELECT * FROM bhobk WHERE CAST(vkey AS DOUBLE) = -5.8543800021562635e-33 AND vvalid = 1;")
#
# print(cur.fetchall())
