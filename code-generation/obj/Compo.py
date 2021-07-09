import pandas as pd
import json
import cv2


def visualize_Compos(compos_html, img):
    board = img.copy()
    for compo in compos_html:
        board = compo.visualize(board)
    cv2.imshow('compos', board)
    cv2.waitKey()
    cv2.destroyWindow('compos')


class Compo:
    def __init__(self, compo_id, compo_class,
                 compo_df=None, children=None, parent=None, img=None, img_shape=None, list_alignment=None):
        self.compo_df = compo_df
        self.compo_id = compo_id
        self.compo_class = compo_class

        # get the clip for single element
        self.compo_clip = compo_df['clip_path'].replace('data/output\clips\\', '') \
            if compo_df is not None and children is None and 'clip_path' in compo_df.index else None
        self.children = children if children is not None else []    # CompoHTML objs
        self.parent = parent                                        # CompoHTML obj
        self.type = 'compo'

        # compo boundary
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.width = None
        self.height = None

        self.img = img
        self.img_shape = img_shape

        self.list_alignment = list_alignment

    def init_boundary(self):
        compo = self.compo_df
        self.top = int(compo['row_min'].min())
        self.left = int(compo['column_min'].min())
        self.bottom = int(compo['row_max'].max())
        self.right = int(compo['column_max'].max())
        self.width = int(self.right - self.left)
        self.height = int(self.bottom - self.top)

    def put_info(self):
        info = {'class': self.compo_class,
                'column_min': self.left, 'column_max': self.right, 'row_min': self.top, 'row_max': self.bottom,
                'height': self.height, 'width': self.width}
        return info

    def add_child(self, child):
        '''
        :param child: CompoHTML object
        '''
        self.children.append(child)
        self.compo_df.append(child.compo_df)
        self.init_boundary()

    def visualize(self, img=None, flag='line', show=False, color=(0,255,0)):
        fill_type = {'line':2, 'block':-1}
        img = self.img if img is None else img
        board = img.copy()
        board = cv2.rectangle(board, (self.left, self.top), (self.right, self.bottom), color, fill_type[flag])
        if show:
            cv2.imshow('compo', board)
            cv2.waitKey()
            cv2.destroyWindow('compo')
        return board
