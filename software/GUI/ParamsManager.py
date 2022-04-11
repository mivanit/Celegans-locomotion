import os
import json
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QGroupBox, QDialog, QPushButton

from software.GUI.ParamEditor import ParamEditor
from software.GUI.Launcher import Launcher
from software.Signal.Signal import MYSIGNAL
from software.Utils.FILE_DEFAULT_NAME import PARAMS_FILE_NAME


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
            MYSIGNAL.Change_param.connect(self._change_one_param)
            if os.path.exists(params_dir):
                with open(params_dir, 'r', encoding='utf-8') as fin:
                    self.params: dict = json.load(fin)
            else:
                with open("input/params.json", 'r', encoding='utf-8') as fin:
                    self.params: dict = json.load(fin)
                with open(params_dir, 'a+') as fout:
                    json.dump(self.params, fout)
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
                editor = ParamEditor(obj_dir=self.params_dir,
                                     nodes=self.nodes + [idx, param_name],
                                     value=value)
                if isinstance(value, str):
                    editor.setEnabled(False)
                layout_line.addWidget(editor)
            layout.addRow(layout_line)
        return layout

    def _change_one_param(self, nodes: list):
        if len(nodes) == 2:
            self.params[nodes[0]] = nodes[1]
        elif len(nodes) == 3:
            self.params[nodes[0]][nodes[1]] = nodes[2]
        elif len(nodes) == 4:
            self.params[nodes[0]][nodes[1]][nodes[2]] = nodes[3]
        elif len(nodes) == 5:
            self.params[nodes[0]][nodes[1]][nodes[2]][nodes[3]] = nodes[4]

    def _update_params(self):
        if not self.nodes:
            with open(self.params_dir, 'w') as fout:
                json.dump(self.params, fout, indent=4)
        self.close()


class CreatorParamsManager(ParamsManager):
    def __init__(self, params_dir, parent=None, nodes=None, params=None, window_size=(800, 650), param_per_row=4):
        super(CreatorParamsManager, self).__init__(params_dir, parent, nodes, params, window_size, param_per_row)

    def _update_params(self):
        if not self.nodes:
            with open(self.params_dir, 'w') as fout:
                json.dump(self.params, fout, indent=4)
            print(self.params_dir[:-12], len(PARAMS_FILE_NAME))
            self.lancher = Launcher(rootdir=self.params_dir[:-12], parent=self)
            self.lancher.show()



if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = ParamsManager("")
    main.show()
    # os.remove("")
    sys.exit(app.exec_())