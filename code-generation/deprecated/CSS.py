import json


class CSS:
    def __init__(self, name, **kwargs):
        self.name = name
        self.attrs = kwargs

        self.css_script = None
        self.generate_css()

    def generate_css(self):
        css = '{\n'
        for a in self.attrs:
            css += '\t' + a + ': ' + self.attrs[a] + ';\n'
        css = css.replace('_', '-')
        self.css_script = self.name + css + '}\n'

    def add_attrs(self, **attrs):
        self.attrs.update(attrs)
        self.generate_css()
