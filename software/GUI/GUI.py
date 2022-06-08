import os

from PyQt5.QtWidgets import QMainWindow, QSizePolicy
from PyQt5 import QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from software.GUI.window import Ui_MainWindow
from software.GUI.MplCanvas import MplCanvas
from software.GUI.LineTable import LineTable
from software.GUI.ParamsManager import ParamsManager, CreatorParamsManager
from software.GUI.MultiDirSelector import MultiDirSelector
from software.Signal.Signal import MYSIGNAL
from software.Utils.FILE_DEFAULT_NAME import PARAMS_FILE_NAME


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        self.actionload.triggered.connect(self.load_file)
        self.actioncreate.triggered.connect(self.create_file)
        self.sc = MplCanvas()
        self.table = LineTable()
        self.table.setMaximumSize(320, 1000)
        toolbar = NavigationToolbar(self.sc, self)
        layout1 = QtWidgets.QVBoxLayout()
        layout1.addWidget(toolbar)
        layout1.addWidget(self.sc)
        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(self.table)
        layout = QtWidgets.QGridLayout()
        layout.addLayout(layout1, 0, 0)
        layout.addLayout(layout2, 0, 1)
        self.centralwidget.setLayout(layout)
        self.centralwidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        MYSIGNAL.Open_Manger.connect(self.open_manager)


    def load_file(self):
        dir_list = MultiDirSelector('D:/Celegans-locomotion/data/').list
        if dir_list:
            for dir_ in dir_list:
                MYSIGNAL.Open_Folder.emit(dir_)

    def create_file(self):
        dir_ = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:', 'D:/Celegans-locomotion/data/',
                                                          QtWidgets.QFileDialog.ShowDirsOnly)
        if dir_ != "":
            params_manager = CreatorParamsManager(params_dir=dir_ + "/" + PARAMS_FILE_NAME, parent=self)
            params_manager.show()

    def open_manager(self, params_dir: str):
        params_manager = ParamsManager(params_dir=params_dir + "/" + PARAMS_FILE_NAME, parent=self)
        params_manager.show()

    def refresh(self, lines):
        self.sc.plot(lines)  # TODO: delete or add
        self.table.refresh(lines)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
