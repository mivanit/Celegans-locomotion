import os
import json
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QGroupBox, QDialog, QPushButton, QScrollArea, QWidget

from software.GUI.ParamEditor import ParamEditor


class ParamsManager(QDialog):
    def __init__(self, params_dir, parent=None, nodes=None, params=None, window_size=(800, 650), param_per_row=4):
        super(ParamsManager, self).__init__()
        self.params_dir = params_dir
        self.parent = parent
        self.param_per_row = param_per_row
        if nodes is None:
            self.nodes = []
            title = self.params_dir
        else:
            title = ""
            self.nodes = nodes
            for i in self.nodes:
                title = title + "_" + str(i)

        if params is None:
            with open(params_dir, 'r') as fin:
                self.params: dict = json.load(fin)
        else:
            self.params = params

        self.resize(window_size[0], window_size[1])
        self.setWindowTitle(title)

        self.sub_manager_list = []
        if isinstance(self.params, list):
            self.setLayout(self._build_table(self.params))
        else:
            self.setLayout(QFormLayout())
            for i in self.params.values():
                if not isinstance(i, dict):
                    self.params = {"": self.params}
                    break
            for group_name, group_params in self.params.items():
                self.layout().addWidget(self._build_group(group_name, group_params))

        self.ok_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        self.layout().addWidget(self.ok_button)
        self.layout().addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self._update_params)
        self.cancel_button.clicked.connect(self.close)

    def _build_group(self, group_name, group_params):
        group = QGroupBox(str(group_name), self)
        idx = 0
        layout = QFormLayout()
        for param_name, value in group_params.items():
            if idx % self.param_per_row == 0:
                layout_line = QHBoxLayout()
                # layout_line.addStretch(2)
            layout_line.addWidget(QLabel(str(param_name)))
            local_node = self.nodes + [group_name, param_name]
            if isinstance(value, dict) or isinstance(value, list):
                expand_button = QPushButton("...")
                self.sub_manager_list.append(ParamsManager(params_dir=self.params_dir,
                                                           parent=self,
                                                           nodes=local_node,
                                                           params=value,
                                                           window_size=(500, 200)))
                expand_button.clicked.connect(self.sub_manager_list[-1].show)
                layout_line.addWidget(expand_button)
            else:
                layout_line.addWidget(ParamEditor(obj_dir=self.params_dir,
                                                  nodes=local_node,
                                                  value=value))
            if idx % self.param_per_row == self.param_per_row - 1 or idx == len(group_params) - 1:
                layout.addRow(layout_line)
            idx += 1
        group.setLayout(layout)
        return group

    def _build_table(self, group_params):
        layout = QFormLayout()
        for idx in range(len(self.params)):
            layout_line = QHBoxLayout()
            for param_name, value in group_params[idx].items():
                layout_line.addWidget(QLabel(str(param_name)))
                layout_line.addWidget(ParamEditor(obj_dir=self.params_dir,
                                                  nodes=self.nodes + [idx, param_name],
                                                  value=value))
            layout.addRow(layout_line)
        return layout

    def _update_params(self):

        self.close()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = ParamsManager("../input/params.json")
    main.show()
    sys.exit(app.exec_())
