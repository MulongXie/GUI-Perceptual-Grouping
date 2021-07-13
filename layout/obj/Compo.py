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
    def __init__(self, compo_id, compo_class,
                 compo_df=None, children=None, parent=None, list_alignment=None):
        self.compo_df = compo_df
        self.compo_id = compo_id
        self.compo_class = compo_class

        # get the clip for single element
        self.compo_clip = compo_df['clip_path'].replace('data/output\clips\\', '') \
            if compo_df is not None and children is None and 'clip_path' in compo_df.index else None
        self.children = children if children is not None else []    # CompoHTML objs
        self.parent = parent                                        # CompoHTML obj
        self.type = 'Compo'

        # compo boundary
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.width = None
        self.height = None

        self.list_alignment = list_alignment
        self.init_boundary()

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
