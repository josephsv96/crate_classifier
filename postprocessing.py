import numpy as np


class TileSplitter:
    """
    To process the output prediction by splitting it into tiles
    """

    def __init__(self, annot_arr):
        self.annot_arr = annot_arr

    @staticmethod
    def Tile_3x3(annot_i, tile_cfg):
        a, b = tile_cfg
        S1 = np.zeros((a, a))
        S2 = np.zeros((a, b))
        S3 = np.zeros((a, a))
        S4 = np.zeros((b, a))
        S5 = np.zeros((b, b))
        S6 = np.zeros((b, a))
        S7 = np.zeros((a, a))
        S8 = np.zeros((a, b))
        S9 = np.zeros((a, a))

        S1 = annot_i[0:a,    0:a,    0]
        S2 = annot_i[0:a,    a:a+b,  0]
        S3 = annot_i[0:a,    a+b:,   0]
        S4 = annot_i[a:a+b,  0:a,    0]
        S5 = annot_i[a:a+b,  a:a+b,  0]
        S6 = annot_i[a:a+b,  a+b:,   0]
        S7 = annot_i[a+b:,   0:a,    0]
        S8 = annot_i[a+b:,   a:a+b,  0]
        S9 = annot_i[a+b:,   a+b:,   0]

        return [S1, S2, S3, S4, S5, S6, S7, S8, S9]
