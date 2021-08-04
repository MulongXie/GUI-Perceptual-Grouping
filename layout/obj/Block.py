import pandas as pd
import cv2
from random import randint as rint

from layout.lib.draw import *

block_id = 0


def slice_blocks(compos, direction='v', border='none'):
    '''
    Vertically or horizontally scan compos
    :param direction: slice vertically or horizontally
    :param compos: Compo objects, including elements and lists
    :param border: block CSS border # solid 2px black
    :return blocks: list of [Block objs]
    :return compos: list of compos not blocked: list of [CompoHTML objects]
    '''
    blocks = []
    block_compos = []
    non_blocked_compos = compos
    global block_id

    is_divided = False
    divider = -1
    prev_divider = 0
    # slice from top to bottom
    if direction == 'v':
        # reverse the direction of next slicing
        next_direction = 'h'
        compos.sort(key=lambda x: x.top)
        for compo in compos:
            # new block
            # if divider is above than this compo's top, then gather the previous block_compos as a block
            if divider < compo.top:
                prev_divider = divider
                divider = compo.bottom

                gap = int(compo.top - prev_divider)
                # gather previous compos in a block
                # a single compo is not be counted as a block
                if len(block_compos) == 1:
                    is_divided = True
                if len(block_compos) > 1:
                    is_divided = True
                    tops = [c.top for c in block_compos]
                    bottoms = [c.bottom for c in block_compos]
                    height = int(max(bottoms) - min(tops))
                    block_id += 1
                    blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))
                    # remove blocked compos
                    non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))
                block_compos = []
            # extend block
            elif compo.top < divider < compo.bottom:
                divider = compo.bottom
            block_compos.append(compo)

        # if there are some sub-blocks, gather the left compos as a block
        if is_divided and len(block_compos) > 1:
            tops = [c.top for c in block_compos]
            bottoms = [c.bottom for c in block_compos]
            height = int(max(bottoms) - min(tops))
            block_id += 1
            blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))
            # remove blocked compos
            non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))

    # slice from left to right
    elif direction == 'h':
        # reverse the direction of next slicing
        next_direction = 'v'
        compos.sort(key=lambda x: x.left)
        for compo in compos:
            # new block
            # if divider is lefter than this compo's right, then gather the previous block_compos as a block
            if divider < compo.left:
                prev_divider = divider
                divider = compo.right

                gap = int(compo.left - prev_divider)
                # gather previous compos in a block
                # a single compo is not to be counted as a block
                if len(block_compos) == 1:
                    is_divided = True
                if len(block_compos) > 1:
                    is_divided = True
                    tops = [c.top for c in block_compos]
                    bottoms = [c.bottom for c in block_compos]
                    height = int(max(bottoms) - min(tops))
                    block_id += 1
                    blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))
                    # remove blocked compos
                    non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))
                block_compos = []
            # extend block
            elif compo.left < divider < compo.right:
                divider = compo.right
            block_compos.append(compo)

        # if there are some sub-blocks, gather the left compos as a block
        if is_divided and len(block_compos) > 1:
            tops = [c.top for c in block_compos]
            bottoms = [c.bottom for c in block_compos]
            height = int(max(bottoms) - min(tops))
            block_id += 1
            blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))
            # remove blocked compos
            non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))

    return blocks, non_blocked_compos


class Block:
    def __init__(self, id, compos,
                 slice_sub_block_direction='h'):
        self.block_id = id
        self.compos = compos                # list of Compo/List objs
        self.sub_blocks = []                # list of Block objs
        self.children = []                  # compos + sub_blocks
        self.compo_class = 'Block'

        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.width = None
        self.height = None

        # slice sub-block comprising multiple compos
        self.sub_blk_alignment = slice_sub_block_direction
        self.slice_sub_blocks()
        self.sort_compos_and_sub_blks()
        # print(self.html_id, slice_sub_block_direction)

        self.init_boundary()

    def init_boundary(self):
        self.top = int(min(self.compos + self.sub_blocks, key=lambda x: x.top).top)
        self.bottom = int(max(self.compos + self.sub_blocks, key=lambda x: x.bottom).bottom)
        self.left = int(min(self.compos + self.sub_blocks, key=lambda x: x.left).left)
        self.right = int(max(self.compos + self.sub_blocks, key=lambda x: x.right).right)
        self.height = int(self.bottom - self.top)
        self.width = int(self.right - self.left)

    def get_inner_compos(self):
        compos = []
        for child in self.children:
            if child.compo_class == 'List':
                compos += child.get_inner_compos()
            else:
                compos.append(child)
        return compos

    def wrap_info(self):
        info = {'id': self.block_id, 'class': 'Block', 'children': [],
                'location': {'left': int(self.left), 'right': int(self.right), 'top': int(self.top), 'bottom': int(self.bottom)}}
        for child in self.children:
            info['children'].append(child.wrap_info())
        return info

    '''
    ******************************
    ********** Children **********
    ******************************
    '''

    def slice_sub_blocks(self):
        '''
        slice the block into sub-blocks
        '''
        self.sub_blocks, self.compos = slice_blocks(self.compos, direction=self.sub_blk_alignment)

    def sort_compos_and_sub_blks(self):
        '''
        combine comps and sub_blocks w.r.t the slicing direction
        :param direction: slicing direction: 'v': from top to bottom; 'h': from left to right
        :return: children: sorted sub-blocks and compos
        '''
        if self.sub_blk_alignment == 'v':
            self.children = sorted(self.compos + self.sub_blocks, key=lambda x: x.top)
        elif self.sub_blk_alignment == 'h':
            self.children = sorted(self.compos + self.sub_blocks, key=lambda x: x.left)

    '''
    ******************************
    ******** Visualization *******
    ******************************
    '''
    def visualize_block(self, img, flag='line', show=False, color=(166,255,100)):
        fill_type = {'line': 2, 'block': -1}
        board = img.copy()
        draw_label(board, [self.left, self.top, self.right, self.bottom], color, text='Block', put_text=True)
        if show:
            cv2.imshow('compo', board)
            cv2.waitKey()
            cv2.destroyWindow('compo')
        return board

    def visualize_compos(self, img, flag='line', show=False, color=(0, 255, 0)):
        board = img.copy()
        for compo in self.compos:
            board = compo.visualize(board, flag, color=color)
        if show:
            cv2.imshow('blk_compos', board)
            cv2.waitKey()
            cv2.destroyWindow('blk_compos')
        return board

    def visualize_sub_blocks(self, img, flag='line', show=False, color=(0, 255, 0)):
        board = img.copy()
        for sub_block in self.sub_blocks:
            board = sub_block.visualize_block(board, flag, color=color)
        if show:
            cv2.imshow('blk_compos', board)
            cv2.waitKey()
            cv2.destroyWindow('blk_compos')
        return board

    def visualize_sub_blocks_and_compos(self, img, recursive=False, show=True):
        board = img.copy()
        board = self.visualize_block(board)
        board = self.visualize_compos(board, color=(0,0,200))
        for sub_block in self.sub_blocks:
            board = sub_block.visualize_block(board, color=(200,200,0))
        if show:
            print('Num of sub_block:%i; Num of element: %i' % (len(self.sub_blocks), len(self.compos)))
            cv2.imshow('sub_blocks', board)
            cv2.waitKey()
            cv2.destroyWindow('sub_blocks')

        if recursive:
            board = img.copy()
            for sub_block in self.sub_blocks:
                board = sub_block.visualize_sub_blocks_and_compos(board, recursive)
        return board
