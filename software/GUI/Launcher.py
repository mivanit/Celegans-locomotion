import os, json, subprocess
from typing import List, Dict, Tuple, Union
from PyQt5.QtWidgets import QFormLayout, QLineEdit, QLabel, QGroupBox, QDialog, QPushButton


from software.Utils.util import mkdir, dump_state, genCmd_singlerun
from software.Utils.FILE_DEFAULT_NAME import PARAMS_FILE_NAME
from software.Signal.Signal import MYSIGNAL

class Launcher(QDialog):
    def __init__(self, rootdir, parent=None):
        super(Launcher, self).__init__()
        self.rootdir = rootdir
        self.parent = parent
        MYSIGNAL.Create_line.connect(self.show)
        self.resize(200, 100)
        self.setWindowTitle("create")
        self.setLayout(QFormLayout())
        self.lineEdit = QLineEdit()
        self.layout().addRow(QLabel("duration(s)"), self.lineEdit)
        self.ok_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        self.layout().addRow(self.ok_button)
        self.layout().addRow(self.cancel_button)
        self.ok_button.clicked.connect(self._update_params)
        self.cancel_button.clicked.connect(self.close)


    def _update_params(self):
        dump_state(locals(), self.rootdir)
        cmd: str = genCmd_singlerun(
            output=self.rootdir+"/",
            params=self.rootdir+"/"+PARAMS_FILE_NAME,
            duration=float(self.lineEdit.text())
        )
        print(cmd)
        # run the process, write stderr and stdout to the log file
        with open(self.rootdir + '/log.txt', 'w') as f_log:
            p = subprocess.Popen(
                cmd,
                stderr=subprocess.STDOUT,
                stdout=f_log,
            )
        self.parent.close()
        self.close()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = Launcher("")
    # main.show()
    MYSIGNAL.Create_line.emit()
    sys.exit(app.exec_())


