import pandas as pd
from obj.Compo import Compo

item_id = 0
tag_map = {'Compo': 'div', 'Text': 'div', 'Block': 'div'}
backgrounds = {'Compo': 'grey', 'Text': 'green', 'Block': 'orange'}


def search_list_item_by_compoid(item_ids, compo_id):
    """
        list_items: dictionary => {id of first compo: ListItem}
    """
    for i in item_ids:
        if compo_id in item_ids[i]:
            return i


def gather_list_items(compos):
    '''
        gather compos into a list item in the same row/column of a same pair(list)
        the reason for this is that some list contain more than 2 items, while the 'pair_to' attr only contains relation of two
    '''
    list_items = {}
    item_ids = {}
    mark = []
    global item_id
    for i in range(len(compos)):
        compo = compos.iloc[i]
        if compo['pair_to'] == -1:
            compos.loc[compo['id'], 'list_item'] = item_id
            item_id += 1
        # new item
        elif compo['id'] not in mark and compo['pair_to'] not in mark:
            compo_paired = compos.loc[compo['pair_to']]

            list_items[item_id] = [compo, compo_paired]
            item_ids[item_id] = [compo['id'], compo['pair_to']]

            compos.loc[compo['id'], 'list_item'] = item_id
            compos.loc[compo['pair_to'], 'list_item'] = item_id
            mark += [compo['id'], compo['pair_to']]
            item_id += 1

        elif compo['id'] in mark and compo['pair_to'] not in mark:
            index = search_list_item_by_compoid(item_ids, compo['id'])
            list_items[index].append(compos.loc[compo['pair_to']])
            item_ids[index].append(compo['pair_to'])

            compos.loc[compo['pair_to'], 'list_item'] = index
            mark.append(compo['pair_to'])

        elif compo['id'] not in mark and compo['pair_to'] in mark:
            index = search_list_item_by_compoid(item_ids, compo['pair_to'])
            list_items[index].append(compos.loc[compo['id']])
            item_ids[index].append(compo['id'])

            compos.loc[compo['id'], 'list_item'] = index
            mark.append(compo['id'])

    compos['list_item'] = compos['list_item'].astype(int)
    return list_items


def gather_lists_by_pair_and_group(compos_df):
    '''
    :param compos_df: type of dataframe
    :return: lists: [list_obj]
             non_list_compos: [compoHTML]
    '''
    lists = []
    non_list_compos = []
    # list type of multiple (multiple compos in each list item) for paired groups
    groups = compos_df.groupby('group_pair').groups
    compo_id = 0
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        lists.append(Compo(compo_id=compo_id, compo_class='list-multi', compo_df=compos_df.loc[groups[i]], list_alignment=compos_df.loc[groups[i][0]]['alignment_in_group']))
        compo_id += 1
        # remove selected compos
        compos_df = compos_df.drop(list(groups[i]))

    # list type of single (single compo in each list item) for non-paired groups
    groups = compos_df.groupby('group').groups
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        lists.append(Compo(compo_id=compo_id, compo_class='list-single', compo_df=compos_df.loc[groups[i]], list_alignment=compos_df.loc[groups[i][0]]['alignment_in_group']))
        compo_id += 1
        # remove selected compos
        compos_df = compos_df.drop(list(groups[i]))

    # not count as list for non-grouped compos
    for i in range(len(compos_df)):
        compo_df = compos_df.iloc[i]
        # fake compo presented by colored div
        compo = Compo(compo_id=compo_id, compo_class=compo_df['class'], compo_df=compo_df)
        compo_id += 1
        non_list_compos.append(compo)
    return lists, non_list_compos

