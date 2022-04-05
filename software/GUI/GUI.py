import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from software.GUI.window import Ui_MainWindow
from software.GUI.MplCanvas import MplCanvas
from software.GUI.LineTable import LineTable
from software.Signal.Signal import MYSIGNAL

class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        self.actionopen.triggered.connect(self.open_file)
        self.sc = MplCanvas()
        self.table = LineTable()
        toolbar = NavigationToolbar(self.sc, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)
        self.widget.setLayout(layout)
        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(self.table)
        self.widget_2.setLayout(layout2)
        # self.checkBox.stateChanged.connect(self._add_line)

    def open_file(self):
        dir_ = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:', 'D:/Celegans-locomotion/data/',
                                                          QtWidgets.QFileDialog.ShowDirsOnly) #TODO: return multiple dir
        if dir_ != "":
        # fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", ,"All Files(*);;Text Files(*.txt)")
            MYSIGNAL.Open_Folder.emit(dir_)
        return dir_


    def refresh(self, lines):
        self.sc.plot(lines)  # TODO: delete or add
        self.table.refresh(lines)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())