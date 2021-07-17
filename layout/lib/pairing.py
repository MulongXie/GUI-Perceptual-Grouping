import pandas as pd
import cv2
import numpy as np
import math

import layout.lib.draw as draw


def match_two_groups(g1, g2, max_pos_bias):
    assert g1.iloc[0]['alignment_in_group'] == g2.iloc[0]['alignment_in_group']
    alignment = g1.iloc[0]['alignment_in_group']
    match_num = 0
    pairs = {}
    for i in range(len(g1)):
        c1 = g1.iloc[i]
        for j in range(len(g2)):
            c2 = g2.iloc[j]
            if alignment == 'h':
                if abs(c1.column_min - c2.column_min) < max_pos_bias:
                    pairs[c1['id']] = c2['id']
                    match_num += 1
                    break
            elif alignment == 'v':
                if abs(c1.row_min - c2.row_min) < max_pos_bias:
                    pairs[c1['id']] = c2['id']
                    match_num += 1
                    break
    if match_num >= min(len(g1), len(g2)):
        for i in pairs:
            g1.loc[i, 'pair_to'] = pairs[i]
            g2.loc[pairs[i], 'pair_to'] = i
        return True
    return False


def match_two_groups_distance(g1, g2):
    assert g1.iloc[0]['alignment_in_group'] == g2.iloc[0]['alignment_in_group']
    alignment = g1.iloc[0]['alignment_in_group']
    pairs = {}
    if alignment == 'h':
        min_side = min(list(g1['height']) + list(g2['height']))
    else:
        min_side = min(list(g1['width']) + list(g2['width']))

    if len(g1) == len(g2):
        distances = []
        for i in range(len(g1)):
            c1 = g1.iloc[i]
            c2 = g2.iloc[i]
            distance = math.sqrt((c1['center_column'] - c2['center_column'])**2 + (c1['center_row'] - c2['center_row'])**2)
            # mismatch if too far
            if distance > min_side * 2:
                return False
            # mismatch if it's too different from others
            if i > 0:
                if max(distance, distances[i-1]) > 2 * min(distance, distances[i-1]):
                    return False
            pairs[c1['id']] = c2['id']
            distances.append(distance)
    else:
        distances = []
        for i in range(len(g1)):
            c1 = g1.iloc[i]
            distance = None
            for j in range(len(g2)):
                c2 = g2.iloc[j]
                d_cur = math.sqrt((c1['center_column'] - c2['center_column']) ** 2 + (c1['center_row'] - c2['center_row']) ** 2)
                if distance is None or distance < d_cur:
                    distance = d_cur
                    pairs[c1['id']] = c2['id']
            # mismatch if too far
            if distance > min_side * 2:
                return False
            # mismatch if it's too different from others
            if i > 0:
                if max(distance, distances[i-1]) > 2 * min(distance, distances[i-1]):
                    return False
            distances.append(distance)
    for i in pairs:
        g1.loc[i, 'pair_to'] = pairs[i]
        g2.loc[pairs[i], 'pair_to'] = i
    return True


def pair_matching_between_multi_groups(groups1, groups2):
    pairs = {}
    pair_id = 0
    for i, g1 in enumerate(groups1):
        for j, g2 in enumerate(groups2):
            if g1.alignment == g2.alignment and abs(g1.compos_number - g2.compos_number) <= 2:
                if match_two_groups(g1, g2, 10):
                    if 'group_pair' not in g1.compos_dataframe.columns:
                        # hasn't paired yet, creat a new pair
                        pair_id += 1
                        g1.compos_dataframe['group_pair'] = pair_id
                        g1.compos_dataframe['group_pair'].astype(int)
                        pairs[pair_id] = [g1, g2]
                    else:
                        # existing group_pair
                        pairs[g1.compos_dataframe.iloc[0]['group_pair']].append(g2)
                    g2.compos_dataframe['group_pair'] = pair_id
                    g2.compos_dataframe['group_pair'].astype(int)
    return pairs


def pair_matching_within_groups(groups, new_pairs=True):
    pairs = {}  # {'pair_id': [dataframe of grouped by certain attr]}
    pair_id = 0
    mark = np.full(len(groups), False)
    if new_pairs:
        for group in groups:
            if 'group_pair' in group.columns:
                group.drop('group_pair', axis=1, inplace=True)
    for i, g1 in enumerate(groups):
        # if mark[i]: continue
        alignment1 = g1.iloc[0]['alignment_in_group']
        for j in range(i + 1, len(groups)):
            g2 = groups[j]
            alignment2 = g2.iloc[0]['alignment_in_group']
            if alignment1 == alignment2:
                if match_two_groups_distance(g1, g2):
                    # print(i, list(g1['group'])[0], mark[i], '-', j, list(g2['group'])[0], mark[j])
                    if not mark[i]:
                        # hasn't paired yet, creat a new pair
                        pair_id += 1
                        g1['group_pair'] = pair_id
                        g2['group_pair'] = pair_id
                        pairs[pair_id] = [g1, g2]
                        mark[i] = True
                        mark[j] = True
                    else:
                        if not mark[j]:
                            g2['group_pair'] = g1.iloc[0]['group_pair']
                            pairs[g1.iloc[0]['group_pair']].append(g2)
                            mark[j] = True
                        # if gi and gj are all already marked in different group_pair, merge the two group_pairs together
                        else:
                            # merge all g2's pairing groups with g1's
                            for g in pairs[g2.iloc[0]['group_pair']]:
                                g['group_pair'] = g1.iloc[0]['group_pair']
                                pairs[g1.iloc[0]['group_pair']].append(g)
                            pairs.pop(g2.iloc[0]['group_pair'])

    merged_pairs = None
    for i in pairs:
        for group in pairs[i]:
            if merged_pairs is None:
                merged_pairs = group
            else:
                merged_pairs = merged_pairs.append(group, sort=False)
    return merged_pairs


def pair_visualization(pairs, img, img_shape, show_method='line'):
    board = img.copy()
    if show_method == 'line':
        for id in pairs:
            pair = pairs[id]
            for p in pair:
                board = draw.visualize(board, p.compos_dataframe, img_shape, attr='group_pair', show=False)
    elif show_method == 'block':
        for id in pairs:
            pair = pairs[id]
            for p in pair:
                board = draw.visualize_fill(board, p.compos_dataframe, img_shape, attr='group_pair', show=False)
    cv2.imshow('pairs', board)
    cv2.waitKey()
    cv2.destroyAllWindows()


# def pair_cvt_df(pairs):
#     df = pd.DataFrame()
#     for i in pairs:
#         pair = pairs[i]
#         for group in pair:
#             df = df.append(group.compos_dataframe, sort=False)
#     # df = df.sort_index()
#     df[list(df.filter(like='group'))] = df[list(df.filter(like='group'))].fillna(-1).astype(int)
#     df['pair'] = df['pair'].fillna(-1).astype(int)
#     return df
