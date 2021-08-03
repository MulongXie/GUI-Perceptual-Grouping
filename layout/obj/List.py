from layout.obj.Compo import Compo
from layout.lib.draw import *

import cv2
import numpy as np
from random import randint as rint


class List(Compo):
    def __init__(self, compo_id, list_class, compo_df, list_alignment=None):
        super().__init__(compo_id=compo_id, compo_class='List', compo_df=compo_df)
        self.list_class = list_class    # single/multi
        self.list_alignment = list_alignment
        self.list_items = []  # list of list_items - list_item: a row in a vertical list / a column in a horizontal list

        self.partition_list_items()

    def get_inner_compos(self):
        return list(np.reshape(self.list_items, (1, -1))[0])

    def partition_list_items(self):
        # each row/column contains multiple compos
        groups = self.compo_df.groupby('list_item').groups
        for i in groups:
            group = list(groups[i])
            item_compos_df = self.compo_df.loc[group]
            list_item = []
            for j in range(len(item_compos_df)):
                item = item_compos_df.iloc[j]
                list_item.append(Compo(compo_id='c-' + str(item['id']), compo_class=item['class'], compo_df=item, in_list=self.compo_id))
            self.list_items.append(list_item)

    def visualize_list(self, img=None, flag='line', show=False):
        board = img.copy()
        for list_item in self.list_items:
            color = random_color()
            for compo in list_item:
                board = compo.visualize(board, flag, color=color)
        board = self.visualize(board, flag)
        if show:
            cv2.imshow('list', board)
            cv2.waitKey()
            cv2.destroyWindow('list')
