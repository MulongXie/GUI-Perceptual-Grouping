import json
import pandas as pd
import numpy as np
import copy
import cv2
from random import randint as rint
from sklearn.cluster import DBSCAN

import layout.lib.repetition_recognition as rep
import layout.lib.draw as draw
import layout.lib.pairing as pairing


class ComposDF:
    def __init__(self, json_file=None, json_data=None, gui_img=None):
        self.json_file = json_file
        self.json_data = json_data if json_data is not None else json.load(open(self.json_file))
        self.compos_json = self.json_data['compos']
        self.compos_dataframe = self.cvt_json_to_df()
        self.img = gui_img

        self.item_id = 0    # id of list item

    '''
    ***********************
    *** Basic Operation ***
    ***********************
    '''
    def copy(self):
        return copy.deepcopy(self)

    def reload_compos(self, json_file=None):
        if json_file is None:
            json_file = self.json_file
        self.json_data = json.load(open(json_file))
        self.compos_json = self.json_data['compos']
        self.compos_dataframe = self.cvt_json_to_df()

    def cvt_json_to_df(self):
        df = pd.DataFrame(columns=['id', 'class', 'column_min', 'column_max', 'row_min', 'row_max',
                                   'height', 'width', 'area', 'center', 'center_column', 'center_row', 'text_content',
                                   'children', 'parent'])
        for i, compo in enumerate(self.compos_json):
            if 'clip_path' in compo:
                compo.pop('clip_path')
            if 'text_content' not in compo:
                compo['text_content'] = None
            if 'position' in compo:
                pos = compo['position']
                compo['column_min'], compo['column_max'] = int(pos['column_min']), int(pos['column_max'])
                compo['row_min'], compo['row_max'] = int(pos['row_min']), int(pos['row_max'])
                compo.pop('position')
            else:
                compo['column_min'], compo['column_max'] = int(compo['column_min']), int(compo['column_max'])
                compo['row_min'], compo['row_max'] = int(compo['row_min']), int(compo['row_max'])
            if 'children' not in compo or compo['children'] is None:
                compo['children'] = None
            else:
                compo['children'] = tuple(compo['children'])
            if 'parent' not in compo:
                compo['parent'] = None
            compo['height'], compo['width'] = int(compo['height']), int(compo['width'])
            compo['area'] = compo['height'] * compo['width']
            compo['center'] = ((compo['column_min'] + compo['column_max']) / 2, (compo['row_min'] + compo['row_max']) / 2)
            compo['center_column'] = compo['center'][0]
            compo['center_row'] = compo['center'][1]

            df.loc[i] = compo
        return df

    def to_csv(self, file):
        self.compos_dataframe.to_csv(file)

    def select_by_class(self, categories, no_parent=False, replace=False):
        df = self.compos_dataframe
        df = df[df['class'].isin(categories)]
        if no_parent:
            df = df[pd.isna(df['parent'])]
        if replace:
            self.compos_dataframe = df
        else:
            return df

    def calc_gap_in_group(self, compos=None):
        '''
        Calculate the gaps between elements in each group
        '''
        if compos is None:
            compos = self.compos_dataframe
        compos['gap'] = -1
        groups = compos.groupby('group').groups
        for i in groups:
            group = groups[i]
            if i != -1 and len(group) > 1:
                group_compos = compos.loc[list(groups[i])]
                # check element's alignment (h or v) in each group
                if 'alignment_in_group' in group_compos:
                    alignment_in_group = group_compos.iloc[0]['alignment_in_group']
                else:
                    alignment_in_group = group_compos.iloc[0]['alignment']
                # calculate element gaps in each group
                if alignment_in_group == 'v':
                    # sort elements vertically
                    group_compos = group_compos.sort_values('center_row')
                    for j in range(len(group_compos) - 1):
                        id = group_compos.iloc[j]['id']
                        compos.loc[id, 'gap'] = group_compos.iloc[j + 1]['row_min'] - group_compos.iloc[j]['row_max']
                else:
                    # sort elements horizontally
                    group_compos = group_compos.sort_values('center_column')
                    for j in range(len(group_compos) - 1):
                        id = group_compos.iloc[j]['id']
                        compos.loc[id, 'gap'] = group_compos.iloc[j + 1]['column_min'] - group_compos.iloc[j]['column_max']

    '''
    ******************
    *** Clustering ***
    ******************
    '''
    def cluster_dbscan_by_attr(self, attr, eps, min_samples=1, show=True, show_method='block'):
        '''
        Cluster elements by attributes using DBSCAN
        '''
        x = np.reshape(list(self.compos_dataframe[attr]), (-1, 1))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        tag = 'cluster_' + attr
        self.compos_dataframe[tag] = clustering.labels_
        self.compos_dataframe[tag].astype(int)
        if show:
            if show_method == 'line':
                self.visualize(gather_attr=tag, name=tag)
            elif show_method == 'block':
                self.visualize_fill(gather_attr=tag, name=tag)

    def cluster_area_by_relational_size(self, show=True, show_method='block'):
        self.compos_dataframe['cluster_area'] = -1
        cluster_id = 0
        for i in range(len(self.compos_dataframe) - 1):
            compo_i = self.compos_dataframe.iloc[i]
            for j in range(i + 1, len(self.compos_dataframe)):
                compo_j = self.compos_dataframe.iloc[j]
                if max(compo_i['area'], compo_j['area']) < min(compo_i['area'], compo_j['area']) * 1.8:
                    if compo_i['cluster_area'] != -1:
                        self.compos_dataframe.loc[compo_j['id'], 'cluster_area'] = compo_i['cluster_area']
                    elif compo_j['cluster_area'] != -1:
                        self.compos_dataframe.loc[compo_i['id'], 'cluster_area'] = compo_j['cluster_area']
                        compo_i = self.compos_dataframe.iloc[i]
                    else:
                        self.compos_dataframe.loc[compo_i['id'], 'cluster_area'] = cluster_id
                        self.compos_dataframe.loc[compo_j['id'], 'cluster_area'] = cluster_id
                        compo_i = self.compos_dataframe.iloc[i]
                        cluster_id += 1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='cluster_area', name='cluster_area')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='cluster_area', name='cluster_area')

    def check_group_by_attr(self, target_attr='cluster_column_min', check_by='height', eps=20, show=True, show_method='block'):
        '''
        Double check the group by additional attribute, using DBSCAN upon the additional attribute to filter out the abnormal element
        @target_attr: gather element groups by this attribute
        @check_by: double check the gathered groups by this attribute
        @eps: EPS for DBSCAN
        '''
        compos = self.compos_dataframe
        groups = compos.groupby(target_attr).groups  # {group name: list of compo ids}
        for i in groups:
            if i != -1 and len(groups[i]) > 2:
                group = groups[i]  # list of component ids in the group
                checking_attr = list(compos.loc[group][check_by])
                # cluster compos height
                clustering = DBSCAN(eps=eps, min_samples=1).fit(np.reshape(checking_attr, (-1, 1)))
                checking_attr_labels = list(clustering.labels_)
                checking_attr_label_count = dict((i, checking_attr_labels.count(i)) for i in checking_attr_labels)  # {label: frequency of label}

                # print(i, checking_attr, checking_attr_labels, checking_attr_label_count)
                for label in checking_attr_label_count:
                    # invalid compo if the compo's height is different from others
                    if checking_attr_label_count[label] < 2:
                        for j, lab in enumerate(checking_attr_labels):
                            if lab == label:
                                compos.loc[group[j], target_attr] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr=target_attr, name=target_attr)
            elif show_method == 'block':
                self.visualize_fill(gather_attr=target_attr, name=target_attr)

    '''
    ****************
    *** Grouping ***
    ****************
    '''
    def group_by_clusters(self, cluster, alignment, show=True, show_method='block'):
        '''
        Group elements by cluster name
        Record element group in 'group' attribute
        '''
        compos = self.compos_dataframe
        if 'group' not in compos.columns:
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
                self.visualize_fill(gather_attr='group', name=name)

    def closer_group_by_mean_area(self, compo_index, group1, group2):
        compo = self.compos_dataframe.loc[compo_index]
        g1 = group1[group1['id'] != compo['id']]
        g2 = group2[group2['id'] != compo['id']]
        # if len(cl2) == 1: return 1
        # elif len(cl1) == 1: return 2

        mean_area1 = g1['area'].mean()
        mean_area2 = g2['area'].mean()

        # if g1 and g2's area is not too different while one of g1 and g2 is length of 1, choose another group
        if abs(mean_area1 - mean_area2) <= 500 or max(mean_area1, mean_area2) < min(mean_area1, mean_area2) * 1.5:
            if len(g1) == 1 and len(g2) > 1:
                return 2
            elif len(g1) > 1 and len(g2) == 1:
                return 1

        # choose the group with similar area
        compo_area = compo['area']
        if abs(compo_area - mean_area1) < abs(compo_area - mean_area2):
            return 1
        return 2

    def group_by_clusters_conflict(self, cluster, alignment, show=True, show_method='block'):
        '''
        If an element is clustered into multiple clusters, assign it to the cluster where the element's area is mostly like others
        Then form the cluster as a group
        '''
        compos = self.compos_dataframe
        group_id = compos['group'].max() + 1

        compo_new_group = {}  # {'id':'new group id'}
        groups = self.compos_dataframe.groupby(cluster).groups
        for i in groups:
            if len(groups[i]) > 1:
                member_num = len(groups[i])
                for j in list(groups[i]):
                    # if the compo hasn't been grouped, then assign it to a new group and no conflict happens
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
                        prev_group = compos[compos['group'] == compos.loc[j, 'group']]
                        if self.closer_group_by_mean_area(j, compos.loc[list(groups[i])], prev_group) == 1:
                            # compos.loc[j, 'group'] = group_id
                            # compos.loc[j, 'alignment'] = alignment
                            compo_new_group[j] = group_id
                        else:
                            member_num -= 1
                group_id += 1

        for i in compo_new_group:
            compos.loc[i, 'group'] = compo_new_group[i]
            compos.loc[i, 'alignment'] = alignment
        self.compos_dataframe['group'].astype(int)

        if show:
            name = cluster if type(cluster) != list else '+'.join(cluster)
            if show_method == 'line':
                self.visualize(gather_attr='group', name=name)
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name=name)

    '''
    *********************************
    *** Similar Group Recognition ***
    *********************************
    '''
    def recognize_similar_blocks(self):
        '''
        Recognize similar Blocks (Containers) that contains similar elements
        '''
        df = self.compos_dataframe
        blocks = df[(df['class']=='Block') & (~pd.isna(df['children'])) & (df['children']!=-1)]
        children_list = []
        connections_list = []
        for i in range(len(blocks)):
            children = df[df['id'].isin(blocks.iloc[i]['children'])].sort_values('center_row')
            # calculate children connections in each block
            children_list.append(children)
            connections_list.append(rep.calc_connections(children))
        # match children connections
        start_pair_id = max(self.compos_dataframe['group_pair']) if 'group_pair' in self.compos_dataframe else 0
        paired_blocks = rep.recog_repetition_block_by_children_connections(children_list, connections_list, start_pair_id)

        # merge the pairing result into the original dataframe
        df_all = self.compos_dataframe
        if paired_blocks is not None:
            if 'group_pair' not in self.compos_dataframe:
                df_all = self.compos_dataframe.merge(paired_blocks, how='left')
            else:
                df_all.loc[df_all[df_all['id'].isin(paired_blocks['id'])]['id'], 'group_pair'] = paired_blocks['group_pair']

            group_id = 0
            # include the parent block in the pair
            children_group = paired_blocks.groupby('parent').groups  # {parent block id: [children id]}
            for i in children_group:
                # set the block as a group
                df_all.loc[i, 'group'] = 'c-' + str(group_id)
                df_all.loc[children_group[i], 'group'] = 'c-' + str(group_id)
                group_id += 1

                df_all.loc[i, 'group_pair'] = df_all.loc[children_group[i][0], 'group_pair']
                # pair to the parent block
                df_all.loc[i, 'pair_to'] = i
                df_all.loc[children_group[i], 'pair_to'] = i

            df_all = df_all.fillna(-1)
            df_all['group_pair'] = df_all['group_pair'].astype(int)
        else:
            if 'group_pair' not in self.compos_dataframe:
                df_all['group_pair'] = -1
        self.compos_dataframe = df_all

    def recognize_element_groups_by_clustering(self, show=False):
        '''
        Recognize repetitive layout of elements that are not contained in Block by clustering
        '''
        df_nontext = rep.recog_repetition_nontext(self, show, only_non_contained_compo=True)
        df_text = rep.recog_repetition_text(self, show, only_non_contained_compo=True)
        df = self.compos_dataframe

        df['alignment'] = np.nan
        df['gap'] = np.nan
        if 'alignment' in df_nontext:
            df.loc[df['alignment'].isna(), 'alignment'] = df_nontext['alignment']
        if 'gap' in df_nontext:
            df.loc[df['gap'].isna(), 'gap'] = df_nontext['gap']
        df = df.merge(df_nontext, how='left')
        if 'alignment' in df_text:
            df.loc[df['alignment'].isna(), 'alignment'] = df_text['alignment']
        if 'gap' in df_text:
            df.loc[df['gap'].isna(), 'gap'] = df_text['gap']
        df = df.merge(df_text, how='left')
        df.rename({'alignment': 'alignment_in_group'}, axis=1, inplace=True)

        # clean and rename attributes
        df = df.fillna(-1)
        df['group'] = -1
        for i in range(len(df)):
            if df.iloc[i]['group_nontext'] != -1:
                df.loc[i, 'group'] = 'nt-' + str(int(df.iloc[i]['group_nontext']))
            elif df.iloc[i]['group_text'] != -1:
                df.loc[i, 'group'] = 't-' + str(int(df.iloc[i]['group_text']))
        groups = df.groupby('group').groups
        for i in groups:
            if len(groups[i]) == 1:
                df.loc[groups[i], 'group'] = -1

        df = df.drop(list(df.filter(like='cluster')), axis=1)
        df = df.drop(columns=['group_nontext', 'group_text'])
        self.compos_dataframe = df

        # slip a group into several ones according to compo gaps if necessary
        self.regroup_compos_by_compos_gap()
        # search and add compos into a group according to compo gaps in the group
        # self.add_missed_compo_to_group_by_gaps(search_outside=False)
        # check group validity by compos gaps in the group, the gaps among compos in a group should be similar
        self.check_group_validity_by_compos_gap()

    '''
    ********************************
    *** Refine Repetition Result ***
    ********************************
    '''
    def regroup_left_compos_by_cluster(self, cluster, alignment, show=True, show_method='block'):
        compos = self.compos_dataframe
        group_id = compos['group'].max() + 1

        # select left compos that in a group only containing itself
        groups = compos.groupby('group').groups
        left_compos_id = []
        for i in groups:
            if i != -1 and len(groups[i]) == 1:
                left_compos_id += list(groups[i])
        left_compos = compos.loc[left_compos_id]

        # regroup the left compos by given cluster
        groups = left_compos.groupby(cluster).groups
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
                self.visualize_fill(gather_attr='group', name=name)

    def regroup_compos_by_compos_gap(self):
        '''
        slip a group into several ones according to compo gaps if necessary
        '''
        self.calc_gap_in_group()
        compos = self.compos_dataframe
        groups = compos.groupby('group').groups  # {group name: list of compo ids}
        for i in groups:
            if i != -1 and len(groups[i]) > 2:
                group = groups[i]  # list of component ids in the group
                group_compos = compos.loc[group]
                if group_compos.iloc[0]['alignment_in_group'] == 'v':
                    group_compos = group_compos.sort_values('center_row')
                else:
                    group_compos = group_compos.sort_values('center_column')
                gaps = list(group_compos['gap'])
                # cluster compos gaps
                clustering = DBSCAN(eps=10, min_samples=1).fit(np.reshape(gaps[:-1], (-1, 1)))
                gap_labels = list(clustering.labels_)   # [lists of labels for different gaps]
                gap_label_count = dict((i, gap_labels.count(i)) for i in gap_labels)  # {label: frequency of label}

                new_group_num = 0
                for label in gap_label_count:
                    if gap_label_count[label] >= 2:
                        # select compos with same gaps to form a new group
                        new_group = pd.DataFrame()
                        for j, lab in enumerate(gap_labels):
                            if lab == label:
                                new_group = new_group.append(compos.loc[group_compos.iloc[j]['id']])
                                self.calc_gap_in_group(new_group)
                        new_gap = list(new_group['gap'])
                        # check if the new group is valid by gaps (should have the same gaps between compos)
                        is_valid_group = True
                        for j in range(1, len(new_gap) - 1):
                            if abs(new_gap[j] - new_gap[j - 1]) > 10:
                                is_valid_group = False
                                break
                        if is_valid_group:
                            # do not change the first group
                            if new_group_num >= 1:
                                compos.loc[new_group['id'], 'group'] += '-' + str(new_group_num)

                                # check the last compo
                                last_compo = compos.loc[group[-1]]
                                if new_group.iloc[-1]['alignment_in_group'] == 'v':
                                    new_group.sort_values('center_row')
                                    gap_with_the_last = last_compo['row_min'] - new_group.iloc[-1]['row_max']
                                else:
                                    new_group.sort_values('center_column')
                                    gap_with_the_last = last_compo['column_min'] - new_group.iloc[-1]['column_max']
                                if abs(gap_with_the_last - new_group.iloc[0]['gap']) < 10:
                                    compos.loc[last_compo['id'], 'group'] += '-' + str(new_group_num)
                            new_group_num += 1

    def search_possible_compo(self, anchor_compo, approximate_gap, direction='next'):
        compos = self.compos_dataframe
        if 'alignment_in_group' in anchor_compo:
            alignment_in_group = anchor_compo['alignment_in_group']
        else:
            alignment_in_group = anchor_compo['alignment']

        if alignment_in_group == 'v':
            # search below
            if direction == 'next':
                approx_row = anchor_compo['row_max'] + approximate_gap + 0.5 * anchor_compo['height']
            # search above
            else:
                approx_row = anchor_compo['row_min'] - (approximate_gap + 0.5 * anchor_compo['height'])
            if approx_row >= self.img.shape[0] or approx_row <= 0: return None

            for i in range(len(compos)):
                compo = compos.iloc[i]
                if max(compo['area'], anchor_compo['area']) < min(compo['area'], anchor_compo['area']) * 3 and\
                        compo['row_min'] < approx_row < compo['row_max'] and \
                        max(compo['column_min'], anchor_compo['column_min']) < min(compo['column_max'], anchor_compo['column_max']):
                    return compo
        else:
            # search right
            if direction == 'next':
                approx_column = anchor_compo['column_max'] + approximate_gap + 0.5 * anchor_compo['width']
            # search left
            else:
                approx_column = anchor_compo['column_min'] - (approximate_gap + 0.5 * anchor_compo['width'])
            if approx_column >= self.img.shape[1] or approx_column <= 0: return None

            for i in range(len(compos)):
                compo = compos.iloc[i]
                if max(compo['area'], anchor_compo['area']) < min(compo['area'], anchor_compo['area']) * 3 and \
                        compo['column_min'] < approx_column < compo['column_max'] and \
                        max(compo['row_min'], anchor_compo['row_min']) < min(compo['row_max'], anchor_compo['row_max']):
                    return compo
        return None

    def add_missed_compo_to_group_by_gaps(self, search_outside=True):
        '''
        search and add compos into a group according to compo gaps in the group
        '''
        self.calc_gap_in_group()
        compos = self.compos_dataframe
        groups = compos.groupby('group').groups  # {group name: list of compo ids}
        for i in groups:
            if i != -1 and len(groups[i]) >= 2:
                group = groups[i]  # list of component ids in the group
                group_compos = compos.loc[group]
                if group_compos.iloc[0]['alignment_in_group'] == 'v':
                    group_compos = group_compos.sort_values('center_row')
                else:
                    group_compos = group_compos.sort_values('center_column')
                gaps = list(group_compos['gap'])

                # cluster compos gaps
                clustering = DBSCAN(eps=10, min_samples=1).fit(np.reshape(gaps[:-1], (-1, 1)))
                gap_labels = list(clustering.labels_)
                gap_label_count = dict((i, gap_labels.count(i)) for i in gap_labels)  # {label: frequency of label}
                # get the most counted label
                max_label = max(gap_label_count.items(), key=lambda k: k[1])  # (label id, count number)
                # if there are more than half compos with that similar gap, find possibly missed compos by gap
                if max_label[1] > len(group) * 0.5:
                    anchor_label = max_label[0]
                    # calculate the mean gap with the
                    mean_gap = 0
                    for k, l in enumerate(gap_labels):
                        if gap_labels[k] == anchor_label:
                            mean_gap += gaps[k]
                    mean_gap = int(mean_gap / max_label[1])

                    for k, gap_label in enumerate(gap_labels):
                        # search possible compo from the compo with a different label
                        if gap_label != anchor_label:
                            possible_compo = self.search_possible_compo(anchor_compo=compos.loc[group_compos.iloc[k]['id']], approximate_gap=mean_gap)
                            if possible_compo is not None:
                                compos.loc[possible_compo['id'], 'group'] = i
                                compos.loc[possible_compo['id'], 'alignment_in_group'] = compos.loc[group_compos.iloc[k]['id'], 'alignment_in_group']

                # search possible compos outside the group
                if search_outside:
                    group_compos = compos.loc[group]
                    if group_compos.iloc[0]['alignment_in_group'] == 'v': group_compos = group_compos.sort_values('center_row')
                    else: group_compos = group_compos.sort_values('center_column')
                    # search previously (left or above)
                    possible_compo = self.search_possible_compo(group_compos.iloc[0], group_compos.iloc[0]['gap'], direction='prev')
                    if possible_compo is not None:
                        compos.loc[possible_compo['id'], 'group'] = i
                        compos.loc[possible_compo['id'], 'alignment_in_group'] = group_compos.iloc[0]['alignment_in_group']
                    # search next (right or below)
                    possible_compo = self.search_possible_compo(group_compos.iloc[-1], group_compos.iloc[0]['gap'], direction='next')
                    if possible_compo is not None:
                        compos.loc[possible_compo['id'], 'group'] = i
                        compos.loc[possible_compo['id'], 'alignment_in_group'] = group_compos.iloc[0]['alignment_in_group']

    '''
    ***********************
    *** Validate Groups ***
    ***********************
    '''
    def check_group_of_two_compos_validity_by_areas(self, show=False, show_method='block'):
        groups = self.compos_dataframe.groupby('group').groups
        for i in groups:
            # if the group only has two elements, check if it's valid by elements' areas
            if i != -1 and len(groups[i]) == 2:
                compos = self.compos_dataframe.loc[groups[i]]
                # if the two are too different in area, mark the group as invalid
                if compos['area'].max() - compos['area'].min() > 500 and compos['area'].max() > compos['area'].min() * 2.2:
                    self.compos_dataframe.loc[groups[i], 'group'] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='group', name='valid-two-compos')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name='valid-two-compos')

    def check_unpaired_group_of_two_compos_validity_by_min_area(self, show=False, show_method='block'):
        groups = self.compos_dataframe.groupby('group').groups
        for i in groups:
            # if the group is unpaired and only has two elements, check if it's valid by elements' areas
            if i != -1 and len(groups[i]) == 2:
                compos = self.compos_dataframe.loc[groups[i]]
                if compos.iloc[0]['group_pair'] == -1:
                    # if the two elements are in vertical alignment, then remove the group
                    if compos.iloc[0]['alignment_in_group'] == 'v':
                        self.compos_dataframe.loc[groups[i], 'group'] = -1
                    # if the two are too different in area, mark the group as invalid
                    elif compos['area'].min() < 150 or compos['area'].max() / compos['area'].min() > 1.5:
                        self.compos_dataframe.loc[groups[i], 'group'] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='group', name='valid-two-compos')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name='valid-two-compos')

    def remove_unpaired_group_of_two_compos(self, show=False, show_method='block'):
        groups = self.compos_dataframe.groupby('group').groups
        for i in groups:
            # if the group is unpaired and only has two elements, remove the group
            if i != -1 and len(groups[i]) == 2:
                compos = self.compos_dataframe.loc[groups[i]]
                if compos.iloc[0]['group_pair'] == -1:
                    self.compos_dataframe.loc[groups[i], 'group'] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='group', name='valid-two-compos')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name='valid-two-compos')

    def check_group_validity_by_compos_gap(self, show=False, show_method='block'):
        '''
        check group validity by compos gaps in the group, the gaps among compos in a group should be similar
        '''
        changed = False
        self.calc_gap_in_group()
        compos = self.compos_dataframe
        groups = compos.groupby('group').groups  # {group name: list of compo ids}
        for i in groups:
            if i == -1: continue
            if len(groups[i]) == 1:
                compos.loc[groups[i][0], 'group'] = -1
            elif len(groups[i]) > 2:
                group = groups[i]  # list of component ids in the group
                group_compos = compos.loc[group]
                if group_compos.iloc[0]['alignment_in_group'] == 'v':
                    group_compos = group_compos.sort_values('center_row')
                else:
                    group_compos = group_compos.sort_values('center_column')
                gaps = list(group_compos['gap'])

                # cluster compos gaps
                eps = 30 if group_compos.iloc[0]['class'] == 'Text' else 10
                clustering = DBSCAN(eps=eps, min_samples=1).fit(np.reshape(gaps[:-1], (-1, 1)))
                gap_labels = list(clustering.labels_)
                gap_label_count = dict((i, gap_labels.count(i)) for i in gap_labels)  # {label: frequency of label}

                for label in gap_label_count:
                    # invalid compo if the compo's gap with others is different from others
                    if gap_label_count[label] < 2:
                        for j, lab in enumerate(gap_labels):
                            if lab == label:
                                compos.loc[group_compos.iloc[j]['id'], 'group'] = -1
                                changed = True
        self.check_group_of_two_compos_validity_by_areas(show=show)

        # recursively run till no changes
        if changed:
            self.check_group_validity_by_compos_gap()

        if show:
            if show_method == 'line':
                self.visualize(gather_attr='group', name='valid')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name='valid')

    def check_unpaired_group_validity_by_interleaving(self):
        compos = self.compos_dataframe
        groups = compos.groupby('group').groups  # {group name: list of compo ids}
        if -1 not in groups:
            return
        ungrouped_compos = groups[-1]  # list of ungrouped compo id
        for i in groups:
            # only check unpaired groups
            if i == -1 or compos.loc[groups[i]].iloc[0]['group_pair'] != -1 or len(groups[i]) > 2: continue
            group_compos = compos.loc[groups[i]]
            group_bound = [group_compos['column_min'].min(), group_compos['row_min'].min(), group_compos['column_max'].max(), group_compos['row_max'].max()]
            for j in ungrouped_compos:
                c = compos.loc[j]
                # intersection area
                left = max(group_bound[0], c['column_min'])
                top = max(group_bound[1], c['row_min'])
                right = min(group_bound[2], c['column_max'])
                bottom = min(group_bound[3], c['row_max'])
                width = max(0, right - left)
                height = max(0, bottom - top)
                # if intersected
                if width == 0 or height == 0:
                    continue
                else:
                    if width * height / c['area'] > 0.7:
                        compos.loc[groups[i], 'group'] = -1

    def remove_invalid_groups(self):
        # self.check_unpaired_group_of_two_compos_validity_by_min_area()
        self.remove_unpaired_group_of_two_compos()
        self.check_unpaired_group_validity_by_interleaving()

    def add_missed_compos_by_checking_group_item(self):
        df = self.compos_dataframe
        pairs = df.groupby('group_pair').groups
        for p in pairs:
            if p == -1: continue
            pair = pairs[p]
            pair_all_compos = df.loc[pair]
            paired_groups = pair_all_compos.groupby('group').groups

            max_group_compo_num = max([len(paired_groups[i]) for i in paired_groups])
            for i in paired_groups:
                # Identify abnormal groups that have fewer compos that others do
                if len(paired_groups[i]) < max_group_compo_num:
                    # calculate the related position of the group compos in their paired item
                    group_compos = df.loc[paired_groups[i]]  # compos in the abnormal group
                    compo_related_pos = pairing.calc_compo_related_position_in_its_paired_item(group_compos, pair_all_compos)  # (column_min, row_min, column_max, row_max)

                    # identify the abnormal item and its position
                    abnormal_items = pair_all_compos[~pair_all_compos['list_item'].isin(list(group_compos['list_item']))]
                    abnormal_items_grp = abnormal_items.groupby('list_item').groups
                    for j in abnormal_items_grp:
                        abnormal_item = abnormal_items.loc[abnormal_items_grp[j]]
                        abnormal_item_pos = abnormal_item['column_min'].min(), abnormal_item['row_min'].min()

                        # calculate the potential missed compo area based on the related compo position and the absolute item position
                        potential_missed_compo_area = (compo_related_pos[0] + abnormal_item_pos[0], compo_related_pos[1] + abnormal_item_pos[1],
                                                       compo_related_pos[2] + abnormal_item_pos[0], compo_related_pos[3] + abnormal_item_pos[1])

                        # find the potential missed compo through iou with the potential_missed_compo_area
                        missed_compo_id = pairing.find_missed_compo_by_iou_with_potential_area(potential_missed_compo_area, df)
                        if missed_compo_id and df.loc[missed_compo_id, 'class'] == group_compos.iloc[0]['class']:
                            # print(df.loc[missed_compo_id, 'class'], group_compos.iloc[0]['class'], i, j)
                            df.loc[missed_compo_id, 'group_pair'] = p
                            df.loc[missed_compo_id, 'group'] = i
                            df.loc[missed_compo_id, 'list_item'] = j

    '''
    ******************************
    ******** Pair Groups *********
    ******************************
    '''
    def pair_groups(self):
        # gather by same groups
        all_groups = self.split_groups('group')

        # pairing between groups
        if 'group_pair' in self.compos_dataframe:
            start_pair_id = max(self.compos_dataframe['group_pair'])
        else:
            start_pair_id = 0
        pairs = pairing.pair_matching_within_groups(all_groups, start_pair_id)
        # merge the pairing result into the original dataframe
        df_all = self.compos_dataframe
        if pairs is not None:
            if 'group_pair' not in self.compos_dataframe:
                df_all = self.compos_dataframe.merge(pairs, how='left')
            else:
                df_all.loc[df_all[df_all['id'].isin(pairs['id'])]['id'], 'group_pair'] = pairs['group_pair']
                df_all.loc[df_all[df_all['id'].isin(pairs['id'])]['id'], 'pair_to'] = pairs['pair_to']

            # add alignment between list items
            # df_all.rename({'alignment': 'alignment_list'}, axis=1, inplace=True)
            # df_all.loc[list(df_all[df_all['alignment_list'] == 'v']['id']), 'alignment_item'] = 'h'
            # df_all.loc[list(df_all[df_all['alignment_list'] == 'h']['id']), 'alignment_item'] = 'v'

            # fill nan and change type
            df_all = df_all.fillna(-1)
            # df_all[list(df_all.filter(like='group'))] = df_all[list(df_all.filter(like='group'))].astype(int)
            df_all['group_pair'] = df_all['group_pair'].astype(int)
            df_all['pair_to'] = df_all['pair_to'].astype(int)
        else:
            df_all['group_pair'] = -1
            df_all['pair_to'] = -1
        self.compos_dataframe = df_all

    def split_groups(self, group_name):
        compos = self.compos_dataframe
        groups = []
        g = compos.groupby(group_name).groups
        for i in g:
            if i == -1 or len(g[i]) <= 1:
                continue
            groups.append(compos.loc[g[i]])
        return groups

    '''
    ******************************
    ******* Item Partition *******
    ******************************
    '''
    def list_item_partition(self):
        '''
        identify list item (paired elements) in each compound large group
        track paired compos' "pair_to" attr to assign "list_item" id
        '''
        if 'pair_to' not in self.compos_dataframe:
            return
        compos = self.compos_dataframe
        groups = compos.groupby("group_pair").groups
        listed_compos = pd.DataFrame()
        for i in groups:
            if i == -1:
                continue
            group = groups[i]
            paired_compos = self.compos_dataframe.loc[list(group)]
            self.gather_list_items(paired_compos)
            listed_compos = listed_compos.append(paired_compos)

        if len(listed_compos) > 0:
            self.compos_dataframe = self.compos_dataframe.merge(listed_compos, how='left')
            self.compos_dataframe['list_item'] = self.compos_dataframe['list_item'].fillna(-1).astype(int)
        else:
            self.compos_dataframe['list_item'] = -1

    def gather_list_items(self, compos):
        '''
            gather compos into a list item in the same row/column of a same pair(list)
            the reason for this is that some list contain more than 2 items, while the 'pair_to' attr only contains relation of two
        '''
        def search_list_item_by_compoid(compo_id):
            """
                list_items: dictionary => {id of first compo: ListItem}
            """
            for i in item_ids:
                if compo_id in item_ids[i]:
                    return i

        list_items = {}
        item_ids = {}
        mark = []
        for i in range(len(compos)):
            compo = compos.iloc[i]
            if compo['pair_to'] == -1:
                compos.loc[compo['id'], 'list_item'] = self.item_id
                self.item_id += 1
            # new item
            elif compo['id'] not in mark and compo['pair_to'] not in mark:
                compo_paired = compos.loc[compo['pair_to']]

                list_items[self.item_id] = [compo, compo_paired]
                item_ids[self.item_id] = [compo['id'], compo['pair_to']]

                compos.loc[compo['id'], 'list_item'] = self.item_id
                compos.loc[compo['pair_to'], 'list_item'] = self.item_id
                mark += [compo['id'], compo['pair_to']]
                self.item_id += 1

            elif compo['id'] in mark and compo['pair_to'] not in mark:
                index = search_list_item_by_compoid(compo['id'])
                list_items[index].append(compos.loc[compo['pair_to']])
                item_ids[index].append(compo['pair_to'])

                compos.loc[compo['pair_to'], 'list_item'] = index
                mark.append(compo['pair_to'])

            elif compo['id'] not in mark and compo['pair_to'] in mark:
                index = search_list_item_by_compoid(compo['pair_to'])
                list_items[index].append(compos.loc[compo['id']])
                item_ids[index].append(compo['id'])

                compos.loc[compo['id'], 'list_item'] = index
                mark.append(compo['id'])

        compos['list_item'] = compos['list_item'].astype(int)
        return list_items

    '''
    *****************************
    ******* Visualization *******
    *****************************
    '''
    def visualize(self, img=None, gather_attr='class', name='board', show=True):
        if img is None:
            img = self.img.copy()
        return draw.visualize(img, self.compos_dataframe, attr=gather_attr, name=name, show=show)

    def visualize_fill(self, img=None, gather_attr='class', name='board', show=True):
        if img is None:
            img = self.img.copy()
        return draw.visualize_fill(img, self.compos_dataframe, attr=gather_attr, name=name, show=show)

    def visualize_cluster(self, show=True):
        board = draw.visualize_group_transparent(self.img.copy(), self.compos_dataframe, 'cluster_center_column', show=show)
        board = draw.visualize_group_transparent(board, self.compos_dataframe, 'cluster_center_row', 0.5, 1, color=(0, 0, 255), show=show)
        return board
