import math
import numpy as np


def check_equal_gap_in_group(compos_df, group_by_attr, pos_anchor):
    grps = compos_df.groupby(group_by_attr)
    max_id = max(grps.groups.keys())
    for k in grps.groups:
        g = grps.groups[k]
        if len(g) == 1:
            continue
        gap_pre = compos_df.loc[g[1]][pos_anchor] - compos_df.loc[g[0]][pos_anchor]
        section = [g[0]]

        for i in range(1, len(g)):
            gap = compos_df.loc[g[i]][pos_anchor] - compos_df.loc[g[i - 1]][pos_anchor]
            # compos shouldn't be in same group as irregular gaps
            if gap_pre > gap * 2 or gap > gap_pre * 2:
                max_id += 1
                compos_df.loc[section, group_by_attr] = max_id
                section = []
            else:
                section.append(g[i])


def recog_repetition_nontext(compos, show=True, only_non_contained_compo=True, inplace=True):
    compos_cp = compos.copy()
    compos_cp.select_by_class(['Compo', 'Background'], no_parent=only_non_contained_compo, replace=True)

    compos_cp.cluster_dbscan_by_attr('center_column', eps=10, show=show, show_method='block')
    compos_cp.cluster_dbscan_by_attr('center_row', eps=10, show=show, show_method='block')
    compos_cp.cluster_dbscan_by_attr('area', eps=500, show=show, show_method='block')

    # check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_center_column', 'row_min')
    # check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_center_row', 'column_min')

    compos_cp.group_by_clusters(cluster=['cluster_area', 'cluster_center_column'], alignment='v', show=show, new_groups=True)
    compos_cp.check_group_of_two_compos_validity_by_areas()
    compos_cp.check_group_validity_by_compos_gap()
    compos_cp.group_by_clusters_conflict(cluster=['cluster_area', 'cluster_center_row'], alignment='h', show=show, show_method='block')
    compos_cp.check_group_of_two_compos_validity_by_areas()
    compos_cp.check_group_validity_by_compos_gap()
    compos_cp.compos_dataframe.rename({'group': 'group_nontext'}, axis=1, inplace=True)

    df = compos_cp.compos_dataframe
    # df = df.drop(columns=['cluster_area', 'cluster_center_column', 'cluster_center_row'])
    return df


def recog_repetition_text(compos, show=True, only_non_contained_compo=True, inplace=True):
    compos_cp = compos.copy()
    compos_cp.select_by_class(['Text'], no_parent=only_non_contained_compo, replace=True)

    compos_cp.cluster_dbscan_by_attr('row_min', 10, show=show, show_method='block')
    compos_cp.cluster_dbscan_by_attr('column_min', 10, show=show, show_method='block')

    # check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_row_min', 'column_min')
    # check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_column_min', 'row_min')

    compos_cp.group_by_clusters('cluster_row_min', alignment='h', new_groups=True, show=show, show_method='block')
    compos_cp.check_group_of_two_compos_validity_by_areas()
    compos_cp.group_by_clusters_conflict('cluster_column_min', alignment='v', show=show, show_method='block')
    compos_cp.check_group_of_two_compos_validity_by_areas()
    compos_cp.regroup_left_compos_by_cluster('cluster_column_min', alignment='v', show=show, show_method='block')
    compos_cp.compos_dataframe.rename({'group': 'group_text'}, axis=1, inplace=True)

    df = compos_cp.compos_dataframe
    # df = df.drop(columns=['cluster_column_min', 'cluster_row_min'])
    return df


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

