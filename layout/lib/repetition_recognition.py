import math
import numpy as np


def recog_repetition_nontext(compos, show=True, only_non_contained_compo=True):
    '''
    produced dataframe attributes: 'alignment', 'group_nontext'
    '''
    compos_cp = compos.copy()
    compos_cp.select_by_class(['Compo', 'Background'], no_parent=only_non_contained_compo, replace=True)
    # if no text compo, return empty dataframe
    if len(compos_cp.compos_dataframe) == 0:
        compos_cp.compos_dataframe['alignment'] = -1
        compos_cp.compos_dataframe['group_nontext'] = -1
        return compos_cp.compos_dataframe

    # step1. cluster compos
    compos_cp.cluster_dbscan_by_attr('center_column', eps=15, show=show)
    compos_cp.cluster_dbscan_by_attr('center_row', eps=15, show=show)
    # compos_cp.cluster_dbscan_by_attr('area', eps=500, show=show)
    compos_cp.cluster_area_by_relational_size(show=show)

    # step2. group compos according to clustering
    compos_cp.group_by_clusters(cluster=['cluster_area', 'cluster_center_column'], alignment='v', show=show)
    compos_cp.check_group_of_two_compos_validity_by_areas(show=show)
    compos_cp.group_by_clusters_conflict(cluster=['cluster_area', 'cluster_center_row'], alignment='h', show=show)
    compos_cp.check_group_of_two_compos_validity_by_areas(show=show)
    compos_cp.compos_dataframe.rename({'group': 'group_nontext'}, axis=1, inplace=True)

    return compos_cp.compos_dataframe


def recog_repetition_text(compos, show=True, only_non_contained_compo=True):
    '''
    produced dataframe attributes: 'alignment', 'group_text'
    '''
    compos_cp = compos.copy()
    compos_cp.select_by_class(['Text'], no_parent=only_non_contained_compo, replace=True)
    # if no text compo, return empty dataframe
    if len(compos_cp.compos_dataframe) == 0:
        compos_cp.compos_dataframe['alignment'] = -1
        compos_cp.compos_dataframe['group_text'] = -1
        return compos_cp.compos_dataframe

    # step1. cluster compos
    compos_cp.cluster_dbscan_by_attr('row_min', 10, show=False)
    compos_cp.check_group_by_attr(target_attr='cluster_row_min', check_by='height', eps=15, show=show)
    compos_cp.cluster_dbscan_by_attr('column_min', 15, show=False)
    compos_cp.check_group_by_attr(target_attr='cluster_column_min', check_by='height', eps=30, show=show)
    
    # step2. group compos according to clustering
    compos_cp.group_by_clusters('cluster_row_min', alignment='h', show=show)
    compos_cp.check_group_of_two_compos_validity_by_areas(show=show)
    compos_cp.group_by_clusters_conflict('cluster_column_min', alignment='v', show=show)
    compos_cp.check_group_of_two_compos_validity_by_areas(show=show)
    compos_cp.regroup_left_compos_by_cluster('cluster_column_min', alignment='v', show=show)
    compos_cp.compos_dataframe.rename({'group': 'group_text'}, axis=1, inplace=True)

    return compos_cp.compos_dataframe


def calc_connections(compos):
    '''
    connection of two compos: (length, id_1, id_2) of the connecting line between two compos' centers
    return: connections between all compos
    '''
    connections = []
    for i in range(len(compos) - 1):
        c1 = compos.iloc[i]
        for j in range(i + 1, len(compos)):
            c2 = compos.iloc[j]
            distance = int(math.sqrt((c1['center_column'] - c2['center_column'])**2 + (c1['center_row'] - c2['center_row'])**2))
            # slope = round((c1['center_row'] - c2['center_row']) / (c1['center_column'] - c2['center_column']), 2)
            connections.append((distance, c1['id'], c2['id']))
    # connections = sorted(connections, key=lambda x: x[0])
    return connections


def match_two_connections(cons1, cons2):
    '''
    input: two lists of connections [(length, id_1, id_2)]
        for a block having n elements, it has n*(n-1)/2 connections (full connection of all nodes)
    '''
    if abs(len(cons1) - len(cons2)) > 1:
        return False
    marked = np.full(len(cons2), False)
    matched_num = 0
    for c1 in cons1:
        for k, c2 in enumerate(cons2):
            # the two connections are matched
            if not marked[k] and max(c1[0], c2[0]) < min(c1[0], c2[0]) * 1.5:
                marked[k] = True
                matched_num += 1
                break
    if matched_num == min(len(cons1), len(cons2)):
        return True
    return False


def recog_repetition_block_by_children_connections(children_list, connections_list, start_pair_id):
    pairs = {}  # {'pair_id': [dataframe of children]}
    pair_id = start_pair_id
    mark = np.full(len(children_list), False)

    for i in range(len(children_list) - 1):
        connections1 = connections_list[i]
        children1 = children_list[i]
        for j in range(i + 1, len(children_list)):
            connections2 = connections_list[j]
            children2 = children_list[j]
            if match_two_connections(connections1, connections2):
                if not mark[i]:
                    # hasn't paired yet, creat a new pair
                    if not mark[j]:
                        pair_id += 1
                        children1['group_pair'] = pair_id
                        children2['group_pair'] = pair_id
                        pairs[pair_id] = [children1, children2]
                        mark[i] = True
                        mark[j] = True
                    # if c2 is already paired, set c1's pair_id as c2's
                    else:
                        children1['group_pair'] = children2.iloc[0]['group_pair']
                        pairs[children2.iloc[0]['group_pair']].append(children1)
                        mark[i] = True
                else:
                    # if c1 is marked while c2 isn't marked
                    if not mark[j]:
                        children2['group_pair'] = children1.iloc[0]['group_pair']
                        pairs[children1.iloc[0]['group_pair']].append(children2)
                        mark[j] = True
                    # if c1 and c2 are all already marked in different group_pair, merge the two group_pairs together
                    else:
                        # merge all g2's pairing groups with g1's
                        if children1.iloc[0]['group_pair'] != children2.iloc[0]['group_pair']:
                            c1_pair_id = children1.iloc[0]['group_pair']
                            c2_pair_id = children2.iloc[0]['group_pair']
                            for c in pairs[c2_pair_id]:
                                c['group_pair'] = c1_pair_id
                                pairs[c1_pair_id].append(c)
                            pairs.pop(c2_pair_id)

    merged_pairs = None
    for i in pairs:
        for children in pairs[i]:
            if merged_pairs is None:
                merged_pairs = children
            else:
                merged_pairs = merged_pairs.append(children, sort=False)
    return merged_pairs

