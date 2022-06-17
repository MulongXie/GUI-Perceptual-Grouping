import pandas as pd
import cv2
import os
import json
import time
from os.path import join as pjoin

import element.detect_text.text_detection as text
import element.detect_compo.ip_region_proposal as ip
import element.detect_merge.merge as merge
from layout.obj.Compos_DF import ComposDF
from layout.obj.Compo import *
from layout.obj.Block import *
from layout.obj.List import *
import layout.lib.draw as draw


class GUI:
    def __init__(self, img_file, compos_json_file=None, output_dir='data/output'):
        self.img_file = img_file
        self.img = cv2.imread(img_file)
        self.img_reshape = self.img.shape
        self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.file_name = img_file.replace('\\', '/').split('/')[-1][:-4]

        self.output_dir = output_dir
        self.ocr_dir = pjoin(self.output_dir, 'ocr') if output_dir is not None else None
        self.non_text_dir = pjoin(self.output_dir, 'ip') if output_dir is not None else None
        self.merge_dir = pjoin(self.output_dir, 'uied') if output_dir is not None else None
        self.layout_dir = pjoin(self.output_dir, 'layout') if output_dir is not None else None

        self.compos_json = None  # {'img_shape':(), 'compos':[]}
        self.compos_df = None    # dataframe for efficient processing
        self.compos = []         # list of Compo objects
        self.detection_result_img = {'text': None, 'non-text': None, 'merge': None}   # visualized detection result

        self.layout_result_img_group = None     # visualize group of compos with repetitive layout
        self.layout_result_img_pair = None      # visualize paired groups
        self.layout_result_img_list = None      # visualize list (paired group) boundary

        self.lists = []     # list of List objects representing lists
        self.blocks = []    # list of Block objects representing blocks

    def save_layout_result_imgs(self):
        os.makedirs(self.layout_dir, exist_ok=True)
        cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-group.jpg'), self.layout_result_img_group)
        cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-pair.jpg'), self.layout_result_img_pair)
        cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-list.jpg'), self.layout_result_img_list)
        # print('Layout recognition result images save to ', output_dir)

    def save_layout_result_json(self):
        os.makedirs(self.layout_dir, exist_ok=True)
        js = []
        for block in self.blocks:
            js.append(block.wrap_info())
        json.dump(js, open(pjoin(self.layout_dir, self.file_name + '.json'), 'w'), indent=4)
        # print('Layout recognition result json save to ', output_dir)

    def save_list(self):
        os.makedirs(self.layout_dir, exist_ok=True)
        js = {'ui': self.file_name, 'list': [], 'multitab': []}
        for lst in self.lists:
            js['list'].append(lst.wrap_list_items())
        json.dump(js, open(pjoin(self.layout_dir, self.file_name + '-list.json'), 'w'), indent=4)

    def save_detection_result(self):
        if not os.path.exists(pjoin(self.merge_dir, self.file_name + '.jpg')):
            os.makedirs(self.ocr_dir, exist_ok=True)
            os.makedirs(self.non_text_dir, exist_ok=True)
            os.makedirs(self.merge_dir, exist_ok=True)
            cv2.imwrite(pjoin(self.ocr_dir, self.file_name + '.jpg'), self.detection_result_img['text'])
            cv2.imwrite(pjoin(self.non_text_dir, self.file_name + '.jpg'), self.detection_result_img['non-text'])
            cv2.imwrite(pjoin(self.merge_dir, self.file_name + '.jpg'), self.detection_result_img['merge'])
        if not os.path.exists(pjoin(self.merge_dir, self.file_name + '.json')):
            json.dump(self.compos_json, open(pjoin(self.merge_dir, self.file_name + '.json'), 'w'), indent=4)

    def save_layout_result(self):
        self.save_detection_result()
        self.save_layout_result_imgs()
        self.save_layout_result_json()
        self.save_list()

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

    def detect_element(self, is_ocr=True, is_non_text=True, is_merge=True, img_resize_longest_side=800, show=False):
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

        key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
                      'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-ui-bar': True}
        if is_ocr:
            self.detection_result_img['text'] = text.text_detection(self.img_file, self.ocr_dir, show=show)
        elif os.path.isfile(pjoin(self.ocr_dir, self.file_name + '.jpg')):
            self.detection_result_img['text'] = cv2.imread(pjoin(self.ocr_dir, self.file_name + '.jpg'))

        if is_non_text:
            self.detection_result_img['non-text'] = ip.compo_detection(self.img_file, self.non_text_dir, key_params, resize_by_height=resize_height, show=show)
        elif os.path.isfile(pjoin(self.non_text_dir, self.file_name + '.jpg')):
            self.detection_result_img['non-text'] = cv2.imread(pjoin(self.non_text_dir, self.file_name + '.jpg'))

        if is_merge:
            os.makedirs(self.merge_dir, exist_ok=True)
            compo_path = pjoin(self.non_text_dir, self.file_name + '.json')
            ocr_path = pjoin(self.ocr_dir, self.file_name + '.json')
            self.detection_result_img['merge'], self.compos_json = merge.merge(self.img_file, compo_path, ocr_path, self.merge_dir, is_remove_bar=True, is_paragraph=True, show=show)

    def load_detection_result(self):
        '''
        Load json detection result from json file
        '''
        self.compos_json = json.load(open(pjoin(self.merge_dir, self.file_name + '.json')))
        self.img_reshape = self.compos_json['img_shape']
        self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.draw_element_detection()

    def load_compos(self, compos):
        '''
        Load compos from objects: {'img_shape':(), 'compos':[]}
        '''
        self.compos_json = compos.copy()
        self.img_reshape = self.compos_json['img_shape']
        self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.draw_element_detection()

    '''
    **************************
    *** Layout Recognition ***
    **************************
    '''
    # *** step1 ***
    def cvt_compos_json_to_dataframe(self):
        '''
        Represent the components using a Pandas DataFrame for the sake of processing
        '''
        self.compos_df = ComposDF(json_data=self.compos_json, gui_img=self.img_resized.copy())

    # *** step2 ***
    def recognize_groups(self):
        '''
        Recognize perceptual groups according to clustering and pairing algorithms
        '''
        # cluster elements into groups according to position and area
        self.compos_df.recognize_element_groups_by_clustering()   # group, alignment_in_group, group_nontext, group_text
        # group similar Blocks (Containers) by checking their children's similarity
        self.compos_df.recognize_similar_blocks()       # group_pair
        # pair clusters (groups) into a larger group
        self.compos_df.pair_groups()                    # group_pair, pair_to
        # identify list item (paired elements) in each compound large group
        self.compos_df.list_item_partition()            # list_item
        # filter out invalid unpaired groups
        self.compos_df.remove_invalid_groups()
        # add missed compos by checking group items
        self.compos_df.add_missed_compos_by_checking_group_item()

    # *** step3 ***
    def cvt_groups_to_list_compos(self):
        '''
        Represent the recognized perceptual groups as List objects
        '''
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
        # regard the list as a type of component in the GUI
        self.compos += self.lists

    # *** step4 ***
    def slice_hierarchical_block(self):
        '''
        Slice the GUI into hierarchical blocks based on the recognized Compos
        '''
        blocks, non_blocked_compos = slice_blocks(self.compos, 'v')
        self.blocks = blocks

    # entry method
    def recognize_layout(self, is_save=True):
        start = time.clock()
        self.cvt_compos_json_to_dataframe()
        self.recognize_groups()
        self.cvt_groups_to_list_compos()
        self.slice_hierarchical_block()
        self.get_layout_result_imgs()
        if is_save:
            self.save_layout_result()
        print("[Layout Recognition Completed in %.3f s] Input: %s Output: %s" % (time.clock() - start, self.img_file, pjoin(self.layout_dir, self.file_name + '.json')))
        # print(time.ctime(), '\n\n')

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
        cv2.imshow('text', cv2.resize(self.detection_result_img['text'], (500, 800)))
        cv2.imshow('non-text', cv2.resize(self.detection_result_img['non-text'], (500, 800)))
        cv2.imshow('merge', cv2.resize(self.detection_result_img['merge'], (500, 800)))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def draw_element_detection(self, line=2):
        board_text = self.img_resized.copy()
        board_nontext = self.img_resized.copy()
        board_all = self.img_resized.copy()
        colors = {'Text':(0,0,255), 'Compo':(0,255,0), 'Block':(0,166,166)}
        for compo in self.compos_json['compos']:
            position = compo['position']
            if compo['class'] == 'Text':
                draw.draw_label(board_text, [position['column_min'], position['row_min'], position['column_max'], position['row_max']], colors[compo['class']], line=line)
            else:
                draw.draw_label(board_nontext, [position['column_min'], position['row_min'], position['column_max'], position['row_max']], colors[compo['class']], line=line)
            draw.draw_label(board_all, [position['column_min'], position['row_min'], position['column_max'], position['row_max']], colors[compo['class']], line=line)

        self.detection_result_img['text'] = board_text
        self.detection_result_img['non-text'] = board_nontext
        self.detection_result_img['merge'] = board_all

    def visualize_layout_recognition(self):
        # self.visualize_all_compos()
        cv2.imshow('group', cv2.resize(self.layout_result_img_group, (500, 800)))
        cv2.imshow('group_pair', cv2.resize(self.layout_result_img_pair, (500, 800)))
        cv2.imshow('list', cv2.resize(self.layout_result_img_list, (500, 800)))
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
            board = lst.visualize_list(board, flag='block')
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

    def visualize_container(self, show=True):
        board = self.img_resized.copy()
        df = self.compos_df.compos_dataframe
        containers = df[df['class'] == 'Block']
        for i in range(len(containers)):
            container = containers.iloc[i]
            children = df.loc[list(container['children'])]
            for j in range(len(children)):
                child = children.iloc[j]
                color = (0,255,0) if child['class'] == 'Compo' else (0,0,255)
                cv2.rectangle(board, (child['column_min'], child['row_min']), (child['column_max'], child['row_max']), color, 2)
            draw.draw_label(board, (container['column_min'], container['row_min'], container['column_max'], container['row_max']), (166,166,0), text='container')
        if show:
            cv2.imshow('container', board)
            cv2.waitKey()
            cv2.destroyWindow('container')
