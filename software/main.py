import sys
from PyQt5.QtWidgets import QApplication

from software.GUI.GUI import MainForm
from software.Lines.LineSet import LineSet
from software.Signal.Signal import MYSIGNAL


class Software:
    def __init__(self):
        self.GUI = MainForm()
        self.lines = LineSet()
        MYSIGNAL.Signal_OneParameter.connect(self.refresh)
        self.GUI.show()

    def refresh(self, folder_dir):
        self.lines.find_lines(folder_dir)
        self.GUI.refresh(self.lines)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Software()
    sys.exit(app.exec_())