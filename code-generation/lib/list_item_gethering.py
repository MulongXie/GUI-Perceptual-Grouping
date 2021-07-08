import pandas as pd
from obj.List import List
from obj.CSS import CSS
from obj.Compo_HTML import CompoHTML

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


def gather_lists_by_pair_and_group(compos):
    '''
    :param compos: type of dataframe
    :return: lists: [list_obj]
             non_list_compos: [compoHTML]
    '''
    lists = []
    non_list_compos = []
    # list type of multiple (multiple compos in each list item) for paired groups
    groups = compos.groupby('group_pair').groups
    list_id = 0
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        lists.append(List(list_id, compos.loc[groups[i]], 'multiple', compos.loc[groups[i][0]]['alignment_in_group']))
        list_id += 1
        # remove selected compos
        compos = compos.drop(list(groups[i]))

    # list type of single (single compo in each list item) for non-paired groups
    groups = compos.groupby('group').groups
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        lists.append(List(list_id, compos.loc[groups[i]], 'single', compos.loc[groups[i][0]]['alignment_in_group']))
        list_id += 1
        # remove selected compos
        compos = compos.drop(list(groups[i]))

    # not count as list for non-grouped compos
    for i in range(len(compos)):
        compo = compos.iloc[i]
        html_id = tag_map[compo['class']] + '-' + str(compo['id'])
        # fake compo presented by colored div
        css = CSS(name='#' + html_id, background=backgrounds[compo['class']], width=str(compo['width']) + 'px', height=str(compo['height']) + 'px')
        compo_html = CompoHTML(compo_class=compo['class'], compo_id=compo['id'], compo_df=compo, html_tag=tag_map[compo['class']], html_id=html_id, css={css.name: css})
        non_list_compos.append(compo_html)
    return lists, non_list_compos

