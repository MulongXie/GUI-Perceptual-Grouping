import pandas as pd
import cv2
from random import randint as rint

from obj.Compo_HTML import CompoHTML
from obj.HTML import HTML
from obj.React import React
from obj.CSS import CSS

block_id = 0


def slice_blocks(compos_html, direction='v', border='none'):
    '''
    Vertically or horizontally scan compos
    :param direction: slice vertically or horizontally
    :param compos_html: CompoHTML objects, including elements and lists
    :param border: block CSS border # solid 2px black
    :return blocks: list of [Block objs]
    :return compos_html: list of compos not blocked: list of [CompoHTML objects]
    '''
    blocks = []
    block_compos = []
    non_blocked_compos = compos_html
    global block_id

    is_divided = False
    divider = -1
    prev_divider = 0
    # slice from top to bottom
    if direction == 'v':
        # reverse the direction of next slicing
        next_direction = 'h'
        compos_html.sort(key=lambda x: x.top)
        for compo in compos_html:
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
                    css_name = '#block-' + str(block_id)
                    # css = CSS(css_name, margin_bottom=str(gap) + 'px', clear='left', border="none")
                    css = CSS(css_name, clear='left', border=border, height=str(height) + 'px')
                    blocks.append(Block(id=block_id, compos=block_compos, slice_sub_block_direction=next_direction,
                                        html_id='block-'+str(block_id), css={css_name: css}))
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
            css_name = '#block-' + str(block_id)
            # css = CSS(css_name, margin_bottom=str(int(block_compos[0].top - prev_divider)) + 'px', clear='left', border="none")
            css = CSS(css_name, clear='left', border=border, height=str(height) + 'px')
            blocks.append(Block(id=block_id, compos=block_compos, slice_sub_block_direction=next_direction,
                                html_id='block-' + str(block_id), css={css_name: css}))
            # remove blocked compos
            non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))

    # slice from left to right
    elif direction == 'h':
        # reverse the direction of next slicing
        next_direction = 'v'
        compos_html.sort(key=lambda x: x.left)
        for compo in compos_html:
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
                    css_name = '#block-' + str(block_id)
                    # css = CSS(css_name, margin_right=str(gap) + 'px', float='left', border="none")
                    css = CSS(css_name, float='left', border=border, height=str(height) + 'px')
                    blocks.append(Block(id=block_id, compos=block_compos, slice_sub_block_direction=next_direction,
                                        html_id='block-' + str(block_id), css={css_name: css}))
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
            css_name = '#block-' + str(block_id)
            # css = CSS(css_name, margin_right=str(int(block_compos[0].left - prev_divider)) + 'px', float='left', border="none")
            css = CSS(css_name, float='left', border=border, height=str(height) + 'px')
            blocks.append(Block(id=block_id, compos=block_compos, slice_sub_block_direction=next_direction,
                                html_id='block-' + str(block_id), css={css_name: css}))
            # remove blocked compos
            non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))

    return blocks, non_blocked_compos


def build_layout_blocks(compos_html, border='none'):
    global block_id
    blocks, single_compos = slice_blocks(compos_html, 'v')
    for compo in single_compos:
        block_id += 1
        css_name = '#block-' + str(block_id)
        css = CSS(css_name, clear='left', border=border, height=str(compo.height) + 'px')
        blocks.append(Block(id=block_id, compos=[compo], slice_sub_block_direction='h',
                            html_id='block-' + str(block_id), css={css_name: css}))

    blocks.sort(key=lambda x: x.top)
    blocks[0].update_css('#' + blocks[0].html_id, margin_top=str(blocks[0].top) + 'px')
    for i in range(1, len(blocks)):
        gap = blocks[i].top - blocks[i - 1].bottom
        blocks[i].update_css('#' + blocks[i].html_id, margin_top=str(gap) + 'px', margin_left=str(blocks[i].left) + 'px')
    return blocks


def visualize_blocks(blocks, img, img_shape):
    board = cv2.resize(img, img_shape)
    for block in blocks:
        board = block.visualize_block(board, show=False)
    cv2.imshow('compos', board)
    cv2.waitKey()
    cv2.destroyWindow('compos')


class Block:
    def __init__(self, id, compos,
                 slice_sub_block_direction='h', html_tag=None, html_id=None, html_class_name=None, css=None):
        self.block_id = id
        self.compos = compos                # list of CompoHTML objs
        self.sub_blocks = []                # list of Block objs
        self.children = []                  # compos + sub_blocks
        self.type = 'block'

        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.width = None
        self.height = None

        # html info
        self.html = None        # HTML obj
        self.html_tag = 'div' if html_tag is None else html_tag
        self.html_id = html_id
        self.html_class_name = html_class_name
        self.html_script = ''   # sting
        self.react = None       # React obj
        self.react_html_script = ''  # string
        self.css = css          # dictionary: {'css-name': CSS obj}
        self.css_script = ''    # string

        # slice sub-block comprising multiple compos
        self.sub_blk_alignment = slice_sub_block_direction
        self.slice_sub_blocks()
        self.sort_compos_and_sub_blks()
        # print(self.html_id, slice_sub_block_direction)

        self.init_boundary()
        self.init_html()
        self.init_css()
        self.init_children_css()
        self.init_react()

    def init_boundary(self):
        self.top = int(min(self.compos + self.sub_blocks, key=lambda x: x.top).top)
        self.bottom = int(max(self.compos + self.sub_blocks, key=lambda x: x.bottom).bottom)
        self.left = int(min(self.compos + self.sub_blocks, key=lambda x: x.left).left)
        self.right = int(max(self.compos + self.sub_blocks, key=lambda x: x.right).right)
        self.height = int(self.bottom - self.top)
        self.width = int(self.right - self.left)

    def init_html(self):
        self.html = HTML(tag=self.html_tag, id=self.html_id, class_name=self.html_class_name)

        for child in self.children:
            self.html.add_child(child.html_script)

        self.html_script = self.html.html_script

    def init_react(self):
        self.react = React(tag=self.html_tag, react_compo_name='Block' + str(self.block_id), id=self.html_id, class_name=self.html_class_name)

        for child in self.children:
            self.react.add_child(child.react_html_script)

        self.react_html_script = self.react.react_html_script

    def init_css(self):
        for sub_block in self.sub_blocks:
            self.css.update(sub_block.css)
        for compo in self.compos:
            self.css.update(compo.css)
        self.css_script = self.css
        self.assembly_css()

    def assembly_css(self):
        self.css_script = ''
        for i in self.css:
            self.css_script += self.css[i].css_script

    def update_css(self, css_name, **attrs):
        if css_name in self.css:
            self.css[css_name].add_attrs(**attrs)
        else:
            self.css[css_name] = CSS(css_name, **attrs)
        self.assembly_css()

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

    def init_children_css(self):
        if self.sub_blk_alignment == 'v':
            for i in range(1, len(self.children)):
                child = self.children[i]
                # if child.type == 'block': continue
                css_name = '#' + child.html_id
                gap = child.top - self.children[i - 1].bottom
                if child.html_tag == 'ul':
                    child.update_css(css_name, padding_top=str(gap) + 'px')
                else:
                    child.update_css(css_name, margin_top=str(gap) + 'px')
                self.css.update(child.css)

        elif self.sub_blk_alignment == 'h':
            for i in range(len(self.children)):
                child = self.children[i]
                # if child.type == 'block': continue
                css_name = '#' + child.html_id
                child.update_css(css_name, float='left')
                if i > 0:
                    gap = child.left - self.children[i - 1].right

                    # for nested compos, cancel float and move upper
                    if gap < 0:
                        self.children[i - 1].update_css('#' + self.children[i - 1].html_id, float='none')
                        gap = child.top - self.children[i - 1].bottom
                        child.update_css(css_name, float='none', margin_top=str(gap) + 'px')
                        continue

                    if child.html_tag == 'ul':
                        child.update_css(css_name, padding_left=str(gap) + 'px', clear='none')
                    else:
                        child.update_css(css_name, margin_left=str(gap) + 'px')
                self.css.update(child.css)
        self.assembly_css()


    '''
    ******************************
    ******** Visualization *******
    ******************************
    '''
    def visualize_block(self, img, flag='line', show=False, color=(0, 255, 0)):
        fill_type = {'line': 2, 'block': -1}
        board = img.copy()
        board = cv2.rectangle(board, (self.left, self.top), (self.right, self.bottom), color, fill_type[flag])
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

    def put_info(self):
        info = {'class':'block',
                'column_min': self.left, 'column_max': self.right, 'row_min':self.top, 'row_max':self.bottom,
                'height': self.height, 'width':self.width}
        if self.html_id is not None:
            info['html_id'] = self.html_id
        if self.html_class_name is not None:
            info['html_class_name'] = self.html_class_name
        return info
