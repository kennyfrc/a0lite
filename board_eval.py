import badgyal
import chess
import torch
import search
import math

CACHE_SIZE = 300000

def q_eval(net, board):
    policy, value = net.eval(board, softmax_temp=1.61)
    return cp_eval(value)

def cp_eval(Q):
    return int(111.7 * math.tan(1.5620688421 * Q))

def eval_board(net, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    squares = {}
    board = chess.Board()
    for rank in [8,7,6,5,4,3,2,1]:
        for file in ['A','B','C','D','E','F','G','H']:
            board.set_fen(fen)
            startpos_eval = q_eval(net, board)

            sq = f"{file}{rank}"
            board.remove_piece_at(getattr(chess, sq))

            score = q_eval(net, board)-startpos_eval
            squares[sq] = score

    return squares

def render(squares,width=5):
    board = "|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|{:>{width}}|\n"
    print(board.format(*squares.values(), width=width))


def main():
    net = badgyal.MGNet(cuda=False)
    squares = eval_board(net)
    render(squares)

main()