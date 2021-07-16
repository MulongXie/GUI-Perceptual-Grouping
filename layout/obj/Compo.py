import pandas as pd
import json
import cv2
import os


def export_compos_as_tree(compos, export_dir='data/output/tree'):
    def build_branch(compo):
        branch = compo.put_info()
        if len(compo.children) > 0:
            branch['children'] = []
            for c in compo.children:
                branch['children'].append(build_branch(c))
        return branch

    os.makedirs(export_dir, exist_ok=True)
    tree = []
    for cp in compos:
        tree.append(build_branch(cp))
    json.dump(tree, open(export_dir + '/tree.json', 'w'), indent=4)
    return tree


def visualize_Compos(compos_html, img):
    board = img.copy()
    for compo in compos_html:
        board = compo.visualize(board)
    cv2.imshow('compos', board)
    cv2.waitKey()
    cv2.destroyWindow('compos')


def cvt_list_and_compos_by_pair_and_group(df):
    '''
    :param df: type of dataframe
    :return: lists: [Compo obj]
             non_list_compos: [Compo obj]
    '''
    lists = []
    non_list_compos = []
    # list type of multiple (multiple compos in each list item) for paired groups
    groups = df.groupby('group_pair').groups
    compo_id = 0
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        lists.append(Compo(compo_id=compo_id, compo_class='List-multi', compo_df=df.loc[groups[i]], list_alignment=df.loc[groups[i][0]]['alignment_in_group']))
        compo_id += 1
        # remove selected compos
        df = df.drop(list(groups[i]))

    # list type of single (single compo in each list item) for non-paired groups
    groups = df.groupby('group').groups
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        lists.append(Compo(compo_id=compo_id, compo_class='List-single', compo_df=df.loc[groups[i]], list_alignment=df.loc[groups[i][0]]['alignment_in_group']))
        compo_id += 1
        # remove selected compos
        df = df.drop(list(groups[i]))

    # not count as list for non-grouped compos
    for i in range(len(df)):
        compo_df = df.iloc[i]
        # fake compo presented by colored div
        compo = Compo(compo_id=compo_id, compo_class=compo_df['class'], compo_df=compo_df)
        compo_id += 1
        non_list_compos.append(compo)
    return lists, non_list_compos


class Compo:
    def __init__(self, compo_id, compo_class, compo_df,
                 children=None, parent=None, list_alignment=None):
        self.compo_df = compo_df   # df can contain one or more elements (list items)
        self.compo_id = compo_id
        self.compo_class = compo_class  # List-multi, List-single, Compo, Text

        # get the clip for single element
        self.children = children if children is not None else []    # CompoHTML objs
        self.parent = parent                                        # CompoHTML obj
        self.type = 'Compo'

        # compo boundary
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.center_row = None
        self.center_column = None
        self.width = None
        self.height = None

        self.list_alignment = list_alignment  # for List
        self.text_content = None  # for Text

        self.init_info()

    def init_info(self):
        compo = self.compo_df
        if self.compo_class in ['List-multi', 'List-single']:
            self.top = int(compo['row_min'].min())
            self.left = int(compo['column_min'].min())
            self.bottom = int(compo['row_max'].max())
            self.right = int(compo['column_max'].max())
            self.center_row = (self.top + self.bottom) / 2
            self.center_column = (self.left + self.right) / 2
            self.width = int(self.right - self.left)
            self.height = int(self.bottom - self.top)
        else:
            self.top, self.left, self.bottom, self.right = compo['row_min'], compo['column_min'], compo['row_max'], compo['column_max']
            self.center_row, self.center_column = compo['center_row'], compo['center_column']
            self.width, self.height = compo['width'], compo['height']
            self.text_content = compo['text_content']
            self.children = compo['children']
            self.parent = compo['parent']

    def put_info(self):
        info = {'class': self.compo_class,
                'column_min': self.left, 'column_max': self.right, 'row_min': self.top, 'row_max': self.bottom,
                'height': self.height, 'width': self.width}
        return info

    def visualize(self, img=None, flag='line', show=False):
        fill_type = {'line':2, 'block':-1}
        color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Text Content': (255, 0, 255)}
        board = img.copy()
        board = cv2.rectangle(board, (self.left, self.top), (self.right, self.bottom), color_map[self.type], fill_type[flag])
        if show:
            cv2.imshow('compo', board)
            cv2.waitKey()
            cv2.destroyWindow('compo')
        return board
