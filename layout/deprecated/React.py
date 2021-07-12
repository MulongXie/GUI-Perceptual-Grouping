import os


def indent(s, no):
    lines = s.split('\n')
    result = ''
    for line in lines:
        result += '\t' * no + line + '\n'
    return result


def assembly_compos_react(compos):
    react_page = 'import React from "react";\n'
    react_compo_names = []
    for compo in compos:
        react_compo_names.append(compo.react.react_compo_name)
        react_page += compo.react.assembly_react_component(compo.css) + '\n'
    react_page += 'export {' + ','.join(react_compo_names) + '};'
    return react_page, react_compo_names


def export_react_program(compos, export_dir='data/output/react'):
    os.makedirs(export_dir, exist_ok=True)
    # 1. block.js
    blocks_js, react_compo_names = assembly_compos_react(compos)
    # 2. index.js
    index_js = 'import React from "react";\nimport ReactDOM from "react-dom";\n' + \
               'import {' + ','.join(react_compo_names) + '} from "./blocks;"\n'
    main_react_script = '\n'.join(['<' + n + '/>' for n in react_compo_names])
    index_js += React(react_compo_name='main', html=main_react_script).assembly_react_component([])
    index_js += "ReactDOM.render(<Main />, document.getElementById('root'))"

    open(os.path.join(export_dir, 'index.js'), 'w').write(index_js)
    open(os.path.join(export_dir, 'blocks.js'), 'w').write(blocks_js)
    return blocks_js, index_js


class React:
    def __init__(self, react_compo_name='', html=None, **kwargs):
        self.attrs = kwargs
        self.react_compo_name = react_compo_name.capitalize()
        self.react_compo_script = ''       # in react component class

        # HTML-React setting
        self.react_html_script = None           # HTML script with react style
        if html is None:
            self.tag = self.init_by_input_attr('tag')
            self.class_name = self.init_by_input_attr('class_name')
            self.id = self.init_by_input_attr('id')
            self.children = self.init_by_input_attr('children', '')  # react script
            self.close = True
            self.generate_react_html()
        else:
            self.react_html_script = html

    def init_by_input_attr(self, attr, non_exist_alt=None):
        if attr in self.attrs:
            return self.attrs[attr]
        return non_exist_alt

    def generate_react_html(self):
        # start
        react = "<" + self.tag
        if self.id is not None:
            if self.react_compo_name is None:
                self.react_compo_name = self.id.replace('-', '').capitalize()
            react += " style={this.state.css." + self.id.replace('-', '') + "}"
        if self.class_name is not None:
            react += " style={this.state.css." + self.class_name.replace('-', '') + "}"
        react += ">\n"

        # body
        if self.children is not None:
            # indent
            children = '\t' + self.children.replace('\n', '\n\t')
            react += children[:-1]

        # close
        if self.close:
            react += "</" + self.tag + ">\n"
        else:
            react[-1] = '/>\n'
        self.react_html_script = react

    def add_child(self, child):
        '''
        :param child: string, html script
        '''
        self.children += child
        self.generate_react_html()

    def assembly_react_component(self, css_dict):
        head = "class " + self.react_compo_name + ' extends React.Component{\n'

        constructor = ''
        if len(css_dict) > 0:
            constructor = 'constructor(){\n' + indent('super();', 1)
            style = 'this.state = {css:{\n'
            for tag in css_dict:
                # refactor css tag
                s = tag.split('-')
                if s[0][0] in ['.', '#']:
                    s[0] = s[0][1:]
                new_tag = ''.join(s)
                # refactor css content
                css = css_dict[tag].css_script
                new_css = ''
                s = css.split('\n')
                for line in s[1:-2]:
                    line = line.replace('\t', '')
                    c = line.split(':')
                    if '-' in c[0]:
                        sp = c[0].split('-')
                        c[0] = sp[0] + sp[1].capitalize()
                    new_css += ':"'.join([c[0], c[1][1:-1]]) + '",\n'
                new_css = new_tag + ':{\n' + indent(new_css[:-2], 1) + '},\n'
                style += new_css
            style = style[:-4] + '\n}'
            constructor += indent(style, 1) + '}},'

        render = 'render(){\n' + \
            indent('return (', 1) + \
            indent(self.react_html_script, 2) + \
            indent(')', 1) + '}'

        react_compo = head + indent(constructor, 1) + indent(render, 1) + '}\n'
        self.react_compo_script = react_compo
        return react_compo
