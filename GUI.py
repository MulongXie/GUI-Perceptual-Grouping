import pandas as pd
import cv2
import os
import json
from os.path import join as pjoin

import element.detect_text.text_detection as text
import element.detect_compo.ip_region_proposal as ip
import element.detect_merge.merge as merge
from layout.obj.Compos_DF import ComposDF
from layout.obj.Compo import *
from layout.obj.Block import *
from layout.obj.List import *


class GUI:
    def __init__(self, img_file, compos_json_file=None, output_dir='data/output'):
        self.img_file = img_file
        self.img = cv2.imread(img_file)
        self.file_name = self.img_file.split('/')[-1][:-4] if img_file is not None else None
        self.img_reshape = self.img.shape
        self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))

        self.output_dir = output_dir
        self.ocr_dir = pjoin(self.output_dir, 'ocr') if output_dir is not None else None
        self.non_text_dir = pjoin(self.output_dir, 'ip') if output_dir is not None else None
        self.merge_dir = pjoin(self.output_dir, 'uied') if output_dir is not None else None

        self.compos_json = None  # {'img_shape':(), 'compos':[]}
        self.compos_df = None    # dataframe for efficient processing
        self.compos = []         # list of Compo objects

        self.detect_result_img_text = None      # visualize text
        self.detect_result_img_non_text = None  # visualize non-text
        self.detect_result_img_merge = None     # visualize all elements

        self.layout_result_img_group = None     # visualize group of compos with repetitive layout
        self.layout_result_img_pair = None      # visualize paired groups
        self.layout_result_img_list = None      # visualize list (paired group) boundary

        self.lists = []     # list of List objects representing lists
        self.blocks = []    # list of Block objects representing blocks

        self.load_compos_from_json(compos_json_file)

    def load_compos_from_json(self, compos_json_file):
        if compos_json_file is not None:
            self.compos_json = json.load(open(compos_json_file))
            self.img_reshape = self.compos_json['img_shape']
            self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))

    def save_result_imgs(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(pjoin(output_dir, self.file_name + '-uied.jpg'), self.detect_result_img_merge)
        cv2.imwrite(pjoin(output_dir, self.file_name + '-group.jpg'), self.layout_result_img_group)
        cv2.imwrite(pjoin(output_dir, self.file_name + '-pair.jpg'), self.layout_result_img_pair)
        cv2.imwrite(pjoin(output_dir, self.file_name + '-list.jpg'), self.layout_result_img_list)

    def save_result_json(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        js = []
        for block in self.blocks:
            js.append(block.wrap_info())
        json.dump(js, open(pjoin(output_dir, self.file_name + '-layout.json'), 'w'), indent=4)

    '''
    *****************************
    *** GUI Element Detection ***
    *****************************
    '''
    def resize_by_longest_side(self, img_resize_longest_side=800):
        height, width = self.img.shape[:2]
        if height > width:
            width_re = int(img_resize_longest_side * (width / height))
            return img_resize_longest_side, width_re, self.img.shape[2]
        else:
            height_re = int(img_resize_longest_side * (height / width))
            return height_re, img_resize_longest_side, self.img.shape[2]

    def element_detection(self, is_ocr=True, is_non_text=True, is_merge=True, img_resize_longest_side=800, show=False):
        if self.img_file is None:
            print('No GUI image is input')
            return
        # resize GUI image by the longest side while detecting non-text elements
        if img_resize_longest_side is not None:
            self.img_reshape = self.resize_by_longest_side(img_resize_longest_side)
            self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
            resize_height = self.img_reshape[0]
        else:
            self.img_reshape = self.img.shape
            self.img_resized = self.img.copy()
            resize_height = None

        key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': False,
                      'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-top-bar': True}
        if is_ocr:
            os.makedirs(self.ocr_dir, exist_ok=True)
            self.detect_result_img_text = text.text_detection(self.img_file, self.ocr_dir, show=show)

        if is_non_text:
            os.makedirs(self.non_text_dir, exist_ok=True)
            self.detect_result_img_non_text = ip.compo_detection(self.img_file, self.non_text_dir, key_params, resize_by_height=resize_height, show=show)

        if is_merge:
            os.makedirs(self.merge_dir, exist_ok=True)
            compo_path = pjoin(self.non_text_dir, self.file_name + '.json')
            ocr_path = pjoin(self.ocr_dir, self.file_name + '.json')
            self.detect_result_img_merge, self.compos_json = merge.merge(self.img_file, compo_path, ocr_path, self.merge_dir, is_remove_top=key_params['remove-top-bar'], show=show)

    '''
    *************************************
    *** Repetitive Layout Recognition ***
    *************************************
    '''
    # convert all compo_df to compo objects
    def cvt_compo_df_to_obj(self):
        df = self.compos_df.compos_dataframe
        self.compos = []
        for i in range(len(df)):
            compo = Compo(df.iloc[i]['id'], df.iloc[i]['class'], df.iloc[i])
            self.compos.append(compo)

    # *** step1 ***
    def cvt_compos_json_to_dataframe(self):
        self.compos_df = ComposDF(json_data=self.compos_json, gui_img=self.img_resized.copy())

    # *** step2 ***
    def recognize_repetitive_layout(self):
        # cluster elements into groups according to position and area
        self.compos_df.repetitive_group_recognition()   # group, alignment_in_group, group_nontext, group_text
        # pair clusters (groups)
        self.compos_df.pair_groups()                    # group_pair, pair_to
        # recognize repetitive block by checking their children's connections
        self.compos_df.repetitive_block_recognition()   # group_pair
        # identify list items in each paired group
        self.compos_df.list_item_partition()            # list_item

    # *** step3 ***
    def cvt_list_and_compos_df_to_obj(self):
        df = self.compos_df.compos_dataframe
        self.lists = []
        self.compos = []

        # multiple list (multiple compos in each list item)
        groups = df.groupby('group_pair').groups
        list_id = 0
        for i in groups:
            if i == -1 or len(groups[i]) == 1:
                continue
            self.lists.append(List(compo_id='l-' + str(list_id), list_class='multi', compo_df=df.loc[groups[i]], list_alignment=df.loc[groups[i][0]]['alignment_in_group']))
            list_id += 1
            # remove selected compos
            df = df.drop(list(groups[i]))

        # single list (single compo in each list item)
        groups = df.groupby('group').groups
        for i in groups:
            if i == -1 or len(groups[i]) == 1:
                continue
            self.lists.append(List(compo_id='l-' + str(list_id), list_class='single', compo_df=df.loc[groups[i]], list_alignment=df.loc[groups[i][0]]['alignment_in_group']))
            list_id += 1
            # remove selected compos
            df = df.drop(list(groups[i]))

        # convert left compos that are not in lists
        for i in range(len(df)):
            compo_df = df.iloc[i]
            self.compos.append(Compo(compo_id='c-' + str(compo_df['id']), compo_class=compo_df['class'], compo_df=compo_df))
        self.compos += self.lists
        # get all compos in lists
        # for lst in self.lists:
        #     self.compos += lst.get_inner_compos()

    # *** step4 ***
    def slice_block(self):
        blocks, non_blocked_compos = slice_blocks(self.compos, 'v')
        self.blocks = blocks

    # entry method
    def layout_recognition(self):
        self.cvt_compos_json_to_dataframe()
        self.recognize_repetitive_layout()
        self.cvt_list_and_compos_df_to_obj()
        self.slice_block()
        self.get_layout_result_imgs()

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def get_layout_result_imgs(self):
        self.layout_result_img_group = self.visualize_compos_df('group', show=False)
        self.layout_result_img_pair = self.visualize_compos_df('group_pair', show=False)
        self.layout_result_img_list = self.visualize_lists(show=False)

    def visualize_element_detection(self):
        cv2.imshow('text', self.detect_result_img_text)
        cv2.imshow('non-text', self.detect_result_img_non_text)
        cv2.imshow('merge', self.detect_result_img_merge)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_layout_recognition(self):
        # self.visualize_all_compos()
        cv2.imshow('group', self.layout_result_img_group)
        cv2.imshow('group_pair', self.layout_result_img_pair)
        cv2.imshow('list', self.layout_result_img_list)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_compos_df(self, visualize_attr, show=True):
        board = self.img_resized.copy()
        return self.compos_df.visualize_fill(board, gather_attr=visualize_attr, name=visualize_attr, show=show)

    def visualize_all_compos(self, show=True):
        board = self.img_resized.copy()
        for compo in self.compos:
            board = compo.visualize(board)
        if show:
            cv2.imshow('compos', board)
            cv2.waitKey()
            cv2.destroyWindow('compos')

    def visualize_lists(self, show=True):
        board = self.img_resized.copy()
        for lst in self.lists:
            board = lst.visualize_list(board)
        if show:
            cv2.imshow('lists', board)
            cv2.waitKey()
            cv2.destroyWindow('lists')
        return board

    def visualize_block(self, block_id, show=True):
        board = self.img_resized.copy()
        self.blocks[block_id].visualize_sub_blocks_and_compos(board, show=show)

    def visualize_blocks(self, show=True):
        board = self.img_resized.copy()
        for block in self.blocks:
            board = block.visualize_block(board)
        if show:
            cv2.imshow('compos', board)
            cv2.waitKey()
            cv2.destroyWindow('compos')
