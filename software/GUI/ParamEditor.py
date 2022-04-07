from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit

from software.Signal.Signal import MYSIGNAL


class ParamEditor(QLineEdit):
    def __init__(self, obj_dir, nodes, value):
        super(ParamEditor, self).__init__()
        self.obj_dir = obj_dir
        self.value = value
        self.nodes = nodes
        self.resize(50, 30)
        self.insert(str(value))
        self.setAlignment(Qt.AlignRight)

