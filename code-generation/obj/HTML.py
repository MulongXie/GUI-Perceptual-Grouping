
class HTML:
    def __init__(self, tag, **kwargs):
        self.tag = tag
        self.attrs = kwargs

        self.class_name = self.init_by_input_attr('class_name')
        self.id = self.init_by_input_attr('id')
        self.children = self.init_by_input_attr('children', '')  # html script
        self.close = True

        self.html_script = None
        self.generate_html()

    def init_by_input_attr(self, attr, non_exist_alt=None):
        if attr in self.attrs:
            return self.attrs[attr]
        return non_exist_alt

    def generate_html(self):
        # start
        html = "<" + self.tag
        if self.id is not None:
            html += " id=\"" + self.id + "\""
        if self.class_name is not None:
            html += " class=\"" + self.class_name + "\""
        html += ">\n"

        # body
        if self.children is not None:
            # indent
            children = '\t' + self.children.replace('\n', '\n\t')
            html += children[:-1]

        # close
        if self.close:
            html += "</" + self.tag + ">\n"
        else:
            html[-1] = '/>\n'
        self.html_script = html

    def add_child(self, child):
        '''
        :param child: string, html script
        '''
        self.children += child
        self.generate_html()
