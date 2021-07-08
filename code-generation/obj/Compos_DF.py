import json
import pandas as pd
import numpy as np
import copy
import cv2
from random import randint as rint
from sklearn.cluster import DBSCAN

import lib.repetition_recognition as rep
import lib.draw as draw
import lib.pairing as pairing
import lib.list_item_gethering as lst


class ComposDF:
    def __init__(self, json_file=None, img_file=None, json_data=None):
        self.json_file = json_file
        self.json_data = json_data if json_data is not None else json.load(open(self.json_file))
        self.compos_json = self.json_data['compos']
        self.compos_dataframe = self.trans_as_df()

        self.img_file = img_file
        self.img = cv2.imread(self.img_file)
        self.img_shape = (self.compos_dataframe.iloc[0].width, self.compos_dataframe.iloc[0].height)

    def copy(self):
        return copy.deepcopy(self)

    def reload_compos(self, json_file=None):
        if json_file is None:
            json_file = self.json_file
        self.json_data = json.load(open(json_file))
        self.compos_json = self.json_data['compos']
        self.compos_dataframe = self.trans_as_df()

    def trans_as_df(self):
        # df = pd.DataFrame(columns=['id', 'column_min', 'column_max', 'row_min', 'row_max',
        #                            'center', 'center_column', 'center_row', 'height', 'width', 'area', 'class', 'clip'])
        df = pd.DataFrame(columns=list(self.compos_json[0].keys()) + ['area', 'center', 'center_column', 'center_row'])
        for i, compo in enumerate(self.compos_json):
            compo['height'], compo['width'] = int(compo['height']), int(compo['width'])
            compo['column_min'], compo['column_max'] = int(compo['column_min']), int(compo['column_max'])
            compo['row_min'], compo['row_max'] = int(compo['row_min']), int(compo['row_max'])

            compo['id'] = i
            compo['area'] = compo['height'] * compo['width']
            compo['center'] = ((compo['column_min'] + compo['column_max']) / 2, (compo['row_min'] + compo['row_max']) / 2)
            compo['center_column'] = compo['center'][0]
            compo['center_row'] = compo['center'][1]
            df.loc[i] = compo
        return df

    def select_by_class(self, categories, replace=False):
        df = self.compos_dataframe
        df = df[df['class'].isin(categories)]
        if replace:
            self.compos_dataframe = df
        else:
            return df

    def visualize(self, gather_attr='class', name='board'):
        draw.visualize(self.img, self.compos_dataframe, self.img_shape, gather_attr, name)

    def visualize_block(self, gather_attr='class', name='board'):
        draw.visualize_block(self.img, self.compos_dataframe, self.img_shape, gather_attr, name)

    def to_csv(self, file):
        self.compos_dataframe.to_csv(file)

    '''
    ******************************
    *** Repetition Recognition ***
    ******************************
    '''
    def repetitive_group_recognition(self, show=False, clean_attrs=True):
        df_nontext = rep.recog_repetition_nontext(self, show)
        df_text = rep.recog_repetition_text(self, show)
        df = self.compos_dataframe

        df = df.merge(df_nontext, how='left')
        df.loc[df['alignment'].isna(), 'alignment'] = df_text['alignment']
        df = df.merge(df_text, how='left')
        df.rename({'alignment': 'alignment_in_group'}, axis=1, inplace=True)

        if clean_attrs:
            df = df.drop(list(df.filter(like='cluster')), axis=1)
            df = df.fillna(-1)

            for i in range(len(df)):
                if df.iloc[i]['group_nontext'] != -1:
                    df.loc[i, 'group'] = 'nt-' + str(int(df.iloc[i]['group_nontext']))
                elif df.iloc[i]['group_text'] != -1:
                    df.loc[i, 'group'] = 't-' + str(int(df.iloc[i]['group_text']))

            groups = df.groupby('group').groups
            for i in groups:
                if len(groups[i]) == 1:
                    df.loc[groups[i], 'group'] = -1
            df.group = df.group.fillna(-1)

        # df = rep.rm_invalid_groups(df)
        self.compos_dataframe = df

    # def cluster_check_correct_by_gap(self, by_attr):

    def cluster_dbscan_by_attr(self, attr, eps, min_samples=1, show=True, show_method='line'):
        x = np.reshape(list(self.compos_dataframe[attr]), (-1, 1))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        tag = 'cluster_' + attr
        self.compos_dataframe[tag] = clustering.labels_
        self.compos_dataframe[tag].astype(int)
        if show:
            if show_method == 'line':
                self.visualize(tag, tag)
            elif show_method == 'block':
                self.visualize_block(tag, tag)

    def cluster_dbscan_by_attrs(self, attrs, eps, min_samples=1, show=True, show_method='line'):
        x = list(self.compos_dataframe[attrs].values)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        tag = 'cluster_' + '_'.join(attrs)
        self.compos_dataframe[tag] = clustering.labels_
        self.compos_dataframe[tag].astype(int)
        if show:
            if show_method == 'line':
                self.visualize(tag, tag)
            elif show_method == 'block':
                self.visualize_block(tag, tag)

    def group_by_clusters(self, cluster, alignment,
                          new_groups=True, show=True, show_method='block'):
        compos = self.compos_dataframe
        if 'group' not in compos.columns or new_groups:
            self.compos_dataframe['group'] = -1
            group_id = 0
        else:
            group_id = compos['group'].max() + 1

        groups = self.compos_dataframe.groupby(cluster).groups
        for i in groups:
            if len(groups[i]) > 1:
                self.compos_dataframe.loc[list(groups[i]), 'group'] = group_id
                self.compos_dataframe.loc[list(groups[i]), 'alignment'] = alignment
                group_id += 1
        self.compos_dataframe['group'].astype(int)

        if show:
            name = cluster if type(cluster) != list else '+'.join(cluster)
            if show_method == 'line':
                self.visualize(gather_attr='group', name=name)
            elif show_method == 'block':
                self.visualize_block(gather_attr='group', name=name)

    def closer_cluster_by_mean_area(self, compo_index, cluster1, cluster2):
        compos = self.compos_dataframe
        compo = compos.loc[compo_index]
        compos = compos[compos['id'] != compo['id']]
        cl1 = compos[compos[cluster1] == compo[cluster1]]
        cl2 = compos[compos[cluster2] == compo[cluster2]]
        if len(cl2) == 1: return 1
        elif len(cl1) == 1: return 2

        mean_area1 = cl1['area'].mean()
        mean_area2 = cl2['area'].mean()

        compo_area = compo['area']
        if abs(compo_area - mean_area1) < abs(compo_area - mean_area2):
            return 1
        return 2

    def closer_cluster_by_mean_distance(self, compo_index, cluster1, cluster2):
        def min_distance(c, cl):
            return np.mean(np.square(abs(cl['center_row'] - c['center_row'])) + np.square(abs(cl['center_column'] - c['center_column'])))
        compos = self.compos_dataframe
        compo = compos.loc[compo_index]
        compos = compos[compos['id'] != compo['id']]
        cl1 = compos[compos[cluster1] == compo[cluster1]]
        cl2 = compos[compos[cluster2] == compo[cluster2]]
        if len(cl2) == 1: return 1
        elif len(cl1) == 1: return 2

        print(min_distance(compo, cl1), min_distance(compo, cl2))
        if min_distance(compo, cl1) < min_distance(compo, cl2):
            return 1
        return 2

    def group_by_clusters_conflict(self, cluster, prev_cluster, alignment, show=True, show_method='block'):
        compos = self.compos_dataframe
        group_id = compos['group'].max() + 1

        groups = self.compos_dataframe.groupby(cluster).groups
        for i in groups:
            if len(groups[i]) > 1:
                member_num = len(groups[i])
                for j in list(groups[i]):
                    if compos.loc[j, 'group'] == -1:
                        compos.loc[j, 'group'] = group_id
                        compos.loc[j, 'alignment'] = alignment
                    # conflict raises if a component can be grouped into multiple groups
                    # then double check it by the average area of the groups
                    else:
                        # keep in the previous group if the it is the only member in a new group
                        if member_num <= 1:
                            continue
                        # close to the current cluster
                        if self.closer_cluster_by_mean_area(j, cluster, prev_cluster) == 1:
                            compos.loc[j, 'group'] = group_id
                            compos.loc[j, 'alignment'] = alignment
                        else:
                            member_num -= 1
                group_id += 1
        self.compos_dataframe['group'].astype(int)

        if show:
            name = cluster if type(cluster) != list else '+'.join(cluster)
            if show_method == 'line':
                self.visualize(gather_attr='group', name=name)
            elif show_method == 'block':
                self.visualize_block(gather_attr='group', name=name)

    '''
    ******************************
    ******** Pair groups *********
    ******************************
    '''
    def split_groups(self, group_name):
        compos = self.compos_dataframe
        groups = []
        g = compos.groupby(group_name).groups
        for i in g:
            if i == -1 or len(g[i]) <= 1:
                continue
            groups.append(compos.loc[g[i]])
        return groups

    def pair_groups(self):
        # gather by same groups
        groups_nontext = self.split_groups('group_nontext')
        groups_text = self.split_groups('group_text')
        all_groups = groups_nontext + groups_text
        # all_groups = self.split_groups('group')

        # pairing between groups
        pairs = pairing.pair_matching_within_groups(all_groups)

        # combine together
        df_all = self.compos_dataframe.merge(pairs, how='left')

        # add alignment between list items
        # df_all.rename({'alignment': 'alignment_list'}, axis=1, inplace=True)
        # df_all.loc[list(df_all[df_all['alignment_list'] == 'v']['id']), 'alignment_item'] = 'h'
        # df_all.loc[list(df_all[df_all['alignment_list'] == 'h']['id']), 'alignment_item'] = 'v'
        df_all = df_all.drop(columns=['group_nontext', 'group_text'])

        # fill nan and change type
        df_all = df_all.fillna(-1)
        # df_all[list(df_all.filter(like='group'))] = df_all[list(df_all.filter(like='group'))].astype(int)
        df_all['group_pair'] = df_all['group_pair'].astype(int)
        df_all['pair_to'] = df_all['pair_to'].astype(int)
        self.compos_dataframe = df_all

    '''
    ******************************
    ******* List Partition *******
    ******************************
    '''
    def list_item_partition(self):
        compos = self.compos_dataframe
        groups = compos.groupby("group_pair").groups
        listed_compos = pd.DataFrame()
        for i in groups:
            if i == -1:
                continue
            group = groups[i]
            paired_compos = self.compos_dataframe.loc[list(group)]
            lst.gather_list_items(paired_compos)
            listed_compos = listed_compos.append(paired_compos)

        self.compos_dataframe = self.compos_dataframe.merge(listed_compos, how='left')
        self.compos_dataframe['list_item'] = self.compos_dataframe['list_item'].fillna(-1).astype(int)
