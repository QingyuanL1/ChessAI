def fen_to_board(fen):
    board = [[None for _ in range(9)] for _ in range(10)]

    x = 0
    y = 9
    for char in fen:
        if char.isdigit():
            x += int(char)
        elif char == '/':
            y -= 1
            x = 0
        else:
            color = 'b' if char.islower() else 'w'
            piece = char.upper()
            board[y][x] = {'piece': piece, 'color': color}
            x += 1

    return board

fen = 'rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR'
board = fen_to_board(fen)
print(board)