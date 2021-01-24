import torch
import badgyal.model as model
import badgyal.net as proto_net
import badgyal.proto.net_pb2 as pb
import chess
from badgyal.board2planes import board2planes, policy2moves, bulk_board2planes
import pylru
import sys
import os.path
from badgyal import WDLNet

CHANNELS=128
BLOCKS=10
SE=4


class JNet(WDLNet):
    def __init__(self, cuda=True, torchScript=False):
        super().__init__(cuda=cuda, torchScript=torchScript)

    def load_net(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        file = os.path.join(my_path, "J104.1-30.pb.gz")
        net = model.Net(CHANNELS, BLOCKS, CHANNELS, SE)
        net.import_proto(file)
        return net