
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


def recog_repetition_nontext(compos, show=True, inplace=True):
    compos_cp = compos.copy()
    compos_cp.select_by_class(['Compo', 'Background'], replace=True)

    compos_cp.cluster_dbscan_by_attr('center_column', eps=5, show=show, show_method='block')
    compos_cp.cluster_dbscan_by_attr('center_row', eps=5, show=show, show_method='block')
    compos_cp.cluster_dbscan_by_attr('area', eps=200, show=show, show_method='block')

    check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_center_column', 'row_min')
    check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_center_row', 'column_min')

    compos_cp.group_by_clusters(cluster=['cluster_area', 'cluster_center_column'], alignment='v', show=show, new_groups=True)
    compos_cp.group_by_clusters(cluster=['cluster_area', 'cluster_center_row'], alignment='h', show=show, new_groups=False, show_method='block')
    compos_cp.compos_dataframe.rename({'group':'group_nontext'}, axis=1, inplace=True)

    df = compos_cp.compos_dataframe
    # df = df.drop(columns=['cluster_area', 'cluster_center_column', 'cluster_center_row'])
    return df


def recog_repetition_text(compos, show=True, inplace=True):
    compos_cp = compos.copy()
    compos_cp.select_by_class(['Text'], replace=True)

    compos_cp.cluster_dbscan_by_attr('row_min', 5, show=show, show_method='line')
    compos_cp.cluster_dbscan_by_attr('column_min', 5, show=show, show_method='line')

    check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_row_min', 'column_min')
    check_equal_gap_in_group(compos_cp.compos_dataframe, 'cluster_column_min', 'row_min')

    compos_cp.group_by_clusters('cluster_row_min', alignment='h', new_groups=True, show=show, show_method='block')
    compos_cp.group_by_clusters_conflict('cluster_column_min', 'cluster_row_min', alignment='v', show=show, show_method='block')
    compos_cp.compos_dataframe.rename({'group':'group_text'}, axis=1, inplace=True)

    df = compos_cp.compos_dataframe
    # df = df.drop(columns=['cluster_column_min', 'cluster_row_min'])
    return df



