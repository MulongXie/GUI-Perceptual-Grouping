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


class GUI:
    def __init__(self, img_file, compos_json_file=None, output_dir='data/output'):
        self.img_file = img_file
        self.img = cv2.imread(img_file)
        self.file_name = self.img_file.split('/')[-1][:-4] if img_file is not None else None
        self.img_reshape = self.img.shape

        self.output_dir = output_dir
        self.ocr_dir = pjoin(self.output_dir, 'ocr') if output_dir is not None else None
        self.non_text_dir = pjoin(self.output_dir, 'ip') if output_dir is not None else None
        self.merge_dir = pjoin(self.output_dir, 'uied') if output_dir is not None else None

        self.compos_json = None  # {'img_shape':(), 'compos':[]}
        self.compos_df = None    # dataframe for efficient processing
        self.compos = []         # list of Compo objects

        self.detect_result_img_text = None       # visualize text
        self.detect_result_img_non_text = None  # visualize non-text
        self.detect_result_img_merge = None     # visualize all elements

        self.lists = []     # list of Compo objects representing List type
        self.blocks = []    # list of Block objects representing Block type

        self.load_compos_from_json(compos_json_file)

    def load_compos_from_json(self, compos_json_file):
        if compos_json_file is not None:
            self.compos_json = json.load(open(compos_json_file))
            self.img_reshape = self.compos_json['img_shape']

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
            resize_height = self.img_reshape[0]
        else:
            self.img_reshape = self.img.shape
            resize_height = None

        key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
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
    # *** step1 ***
    def cvt_compos_json_to_dataframe(self):
        self.compos_df = ComposDF(json_data=self.compos_json)

    # *** step2 ***
    def recognize_repetitive_layout(self):
        # cluster elements into groups according to position and area
        self.compos_df.repetitive_group_recognition()   # group, alignment_in_group, group_nontext, group_text
        # pair clusters (groups)
        self.compos_df.pair_groups()                    # group_pair, pair_to
        # identify list items in each paired group
        self.compos_df.list_item_partition()            # list_item

    # *** step3 ***
    def cvt_compos_df_to_obj(self):
        lists, non_list_compos = cvt_list_and_compos_by_pair_and_group(self.compos_df.compos_dataframe)
        self.lists = lists
        self.compos = lists + non_list_compos

    # *** step4 ***
    def slice_block(self):
        blocks, non_blocked_compos = slice_blocks(self.compos, 'v')
        self.blocks = blocks

    # entry method
    def layout_recognition(self):
        if self.compos_df is None:
            self.cvt_compos_json_to_dataframe()
        self.recognize_repetitive_layout()
        self.cvt_compos_df_to_obj()
        self.slice_block()

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def visualize_element_detection(self):
        cv2.imshow('text', self.detect_result_img_text)
        cv2.imshow('non-text', self.detect_result_img_non_text)
        cv2.imshow('merge', self.detect_result_img_merge)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_layout_recognition(self):
        self.visualize_compos_df('group')
        self.visualize_compos_df('group_pair')
        self.visualize_all_compos()

    def visualize_compos_df(self, visualize_attr):
        board = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.compos_df.visualize_fill(board, gather_attr=visualize_attr)

    def visualize_all_compos(self):
        board = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        for compo in self.compos:
            board = compo.visualize(board)
        cv2.imshow('compos', board)
        cv2.waitKey()
        cv2.destroyWindow('compos')

    def visualize_block(self, block_id):
        board = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.blocks[block_id].visualize_sub_blocks_and_compos(board, show=True)

    def visualize_blocks(self):
        board = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        for block in self.blocks:
            board = block.visualize_block(board)
        cv2.imshow('compos', board)
        cv2.waitKey()
        cv2.destroyWindow('compos')
