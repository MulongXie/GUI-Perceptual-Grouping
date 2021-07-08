import cv2
import numpy as np
from random import randint as rint

colors = {}


def random_color():
    return rint(0, 255), rint(0, 255), rint(0, 255)


def visualize(img, compos_df, resize_shape=None, attr='class', name='board', show=True):
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape)

    board = img.copy()
    for i in range(len(compos_df)):
        compo = compos_df.iloc[i]
        board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), (255, 0, 0))
        board = cv2.putText(board, str(compo[attr]), (compo.column_min + 5, compo.row_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    if show:
        cv2.imshow(name, board)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return board


def visualize_block(img, compos_df, resize_shape=None, attr='class', name='board', show=True):
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape)

    board = img.copy()
    for i in range(1, len(compos_df)):
        compo = compos_df.iloc[i]
        if compo[attr] == -1:
            # board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), random_color(), -1)
            continue
        else:
            # compo[attr] = compo[attr].replace('nt', 'c')
            if compo[attr] not in colors:
                colors[compo[attr]] = random_color()
        board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), colors[compo[attr]], -1)
        board = cv2.putText(board, str(compo[attr]), (compo.column_min + 5, compo.row_min + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    if show:
        cv2.imshow(name, board)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return board
