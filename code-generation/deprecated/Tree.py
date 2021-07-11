import os
import json


def build_branch(compo):
    branch = compo.put_info()
    if len(compo.children) > 0:
        branch['children'] = []
        for c in compo.children:
            branch['children'].append(build_branch(c))
    return branch


def export_tree(compos, export_dir='data/output/tree'):
    os.makedirs(export_dir, exist_ok=True)
    tree = []
    for compo in compos:
        tree.append(build_branch(compo))
    json.dump(tree, open(export_dir + '/tree.json', 'w'), indent=4)
    return tree