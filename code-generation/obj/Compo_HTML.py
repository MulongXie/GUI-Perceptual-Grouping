import pandas as pd
import json
import cv2

from obj.CSS import CSS
from obj.HTML import HTML
from obj.React import React


def visualize_CompoHTMLs(compos_html, img):
    board = img.copy()
    for compo in compos_html:
        board = compo.visualize(board)
    cv2.imshow('compos', board)
    cv2.waitKey()
    cv2.destroyWindow('compos')


class CompoHTML:
    def __init__(self, compo_id, html_tag, compo_class,
                 compo_df=None, html_id=None, html_class_name=None, children=None, parent=None, img=None, img_shape=None, css=None):
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

        # html info
        self.html = None                             # HTML obj
        self.html_id = html_id
        self.html_class_name = html_class_name
        self.html_tag = html_tag
        self.html_tag_map = {'Compo': 'div', 'Text': 'div', 'Block': 'div'}
        self.html_script = ''                        # string
        self.react = None                            # React
        self.react_html_script = ''                       # string
        self.css = {} if css is None else css        # directory of CSS objs, {'.class'/'#id' : CSS obj}

        self.img = img
        self.img_shape = img_shape

        self.init_html()
        self.init_react()
        self.init_boundary()

    def init_html(self):
        self.html = HTML(tag=self.html_tag, id=self.html_id, class_name=self.html_class_name)
        if self.compo_clip is not None:
            clip_img = '<img src="' + self.compo_clip + '" class="clip_img">\n'
            self.html.add_child(clip_img)
        if type(self.children) is not list:
            self.children = [self.children]
        for child in self.children:
            self.html.add_child(child.html_script)
        self.html_script = self.html.html_script

    def init_react(self):
        self.react = React(tag=self.html_tag, id=self.html_id, class_name=self.html_class_name)
        for child in self.children:
            self.react.add_child(child.react_html_script)
        self.react_html_script = self.react.react_html_script

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
        if self.html_id is not None:
            info['html_id'] = self.html_id
        if self.html_class_name is not None:
            info['html_class_name'] = self.html_class_name
        return info

    def add_child(self, child):
        '''
        :param child: CompoHTML object
        '''
        self.children.append(child)
        self.html.add_child(child.html_script)
        self.html_script = self.html.html_script

        self.compo_df.append(child.compo_df)
        self.init_boundary()

    def update_css(self, css_name, **attrs):
        if css_name in self.css:
            self.css[css_name].add_attrs(**attrs)
        else:
            self.css[css_name] = CSS(css_name, **attrs)

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
