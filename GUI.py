import pandas as pd
import cv2
import os
from os.path import join as pjoin

import element.detect_text.text_detection as text
import element.detect_compo.ip_region_proposal as ip
import element.detect_merge.merge as merge


class GUI:
    def __init__(self, img_file=None, output_dir='data/output'):
        self.img_file = img_file
        self.img = cv2.imread(img_file) if img_file is not None else None
        self.file_name = self.img_file.split('/')[-1][:-4] if img_file is not None else None

        self.output_dir = output_dir
        self.ocr_dir = pjoin(self.output_dir, 'ocr') if output_dir is not None else None
        self.non_text_dir = pjoin(self.output_dir, 'ip') if output_dir is not None else None
        self.merge_dir = pjoin(self.output_dir, 'uied') if output_dir is not None else None

        self.compos_json = None
        self.compos_df = None
        self.compos = None

        self.detect_result_img_ocr = None
        self.detect_result_img_non_text = None
        self.detect_result_img_merge = None

    def resize_by_longest_side(self, img_resize_longest_side=800):
        height, width = self.img.shape[:2]
        if height > width:
            width_re = int(img_resize_longest_side * (width / height))
            return width_re, img_resize_longest_side
        else:
            height_re = int(img_resize_longest_side * (height / width))
            return img_resize_longest_side, height_re

    def detect_elements(self, is_ocr=True, is_non_text=True, is_merge=True, img_resize_longest_side=800):
        if self.img_file is None:
            print('No GUI image is input')
            return
        key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
                      'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-top-bar': True}
        if is_ocr:
            os.makedirs(self.ocr_dir, exist_ok=True)
            text.text_detection(self.img_file, self.ocr_dir, show=False)

        if is_non_text:
            os.makedirs(self.non_text_dir, exist_ok=True)
            ip.compo_detection(self.img_file, self.non_text_dir, key_params, resize_by_height=self.resize_by_longest_side(img_resize_longest_side)[1], show=False)

        if is_merge:
            os.makedirs(self.merge_dir, exist_ok=True)
            compo_path = pjoin(self.non_text_dir, self.file_name + '.json')
            ocr_path = pjoin(self.ocr_dir, self.file_name + '.json')
            compos_json = merge.merge(self.img_file, compo_path, ocr_path, self.merge_dir, is_remove_top=key_params['remove-top-bar'], show=True)
            return compos_json
