from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit

from software.Signal.Signal import MYSIGNAL


class ParamEditor(QLineEdit):
    def __init__(self, obj_dir, nodes:list, value):
        super(ParamEditor, self).__init__()
        self.obj_dir = obj_dir
        self.value = value
        self.type = type(value)
        self.nodes = nodes
        self.resize(50, 30)
        self.insert(str(value))
        self.setAlignment(Qt.AlignRight)
        self.textChanged.connect(self._emit_changes)

    def _emit_changes(self):
        self.value = self.type(self.text())
        MYSIGNAL.Change_param.emit(self.nodes + [self.value])

