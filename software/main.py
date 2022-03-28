import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from GUI import main as GUI
from Lines import HeadLine

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure()#figsize=(width, height), dpi=dpi
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class Software:
    def __init__(self):
        self.window = QMainWindow()
        self.GUI = GUI.Ui_MainWindow()
        self.GUI.setupUi(self.window)
        l1 = HeadLine.HeadLine()
        l1.data = [[0, 1, 2, 3, 4], [10, 1, 20, 3, 40]]
        self.line_list = [l1]
        self.GUI.checkBox.setText("line1")
        self.GUI.checkBox_2.setText("line2")

        self.GUI.checkBox.stateChanged.connect(self._add_line)

        fig = plt.Figure()
        # fig.add_axes([0.05, 0.1, 0.8, 0.8])
        plt.plot(l1.data[0],l1.data[1])
        sc=FigureCanvasQTAgg(fig)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(sc, self.window)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)
        self.GUI.widget.setLayout(layout)
        self.window.show()


    def _add_line(self):
        self.line_list[0].visible = True
        print("****")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Software()
    sys.exit(app.exec_())