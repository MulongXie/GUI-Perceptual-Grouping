import cv2
import numpy as np
from random import randint as rint

colors = {}


def random_color():
    return rint(0, 255), rint(0, 255), rint(0, 255)


def draw_label(img, bound, color, text=None, put_text=True):
    cv2.rectangle(img, (bound[0], bound[1]), (bound[2], bound[3]), color, 2)
    if put_text:
        # put text with rectangle
        (w,h),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (bound[0], bound[1] - 20), (bound[0] + w, bound[1] - 20 + h), color, -1)
        cv2.putText(img, text, (bound[0] + 3, bound[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


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


def visualize_fill(img, compos_df, resize_shape=None, attr='class', name='board', show=True):
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape)

    board = img.copy()
    for i in range(len(compos_df)):
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
        cv2.destroyWindow(name)
    return board
