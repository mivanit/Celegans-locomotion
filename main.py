from software.GUI.GUI import MainForm
from software.Lines.LineSet import LineSet
from software.Signal.Signal import MYSIGNAL


class Software:
    def __init__(self):
        self.GUI = MainForm()
        self.lines = LineSet()
        MYSIGNAL.Open_Folder.connect(self._refresh)
        MYSIGNAL.Visibility_Change.connect(self._change_visibility)
        MYSIGNAL.Color_Change.connect(self._change_color)
        self.GUI.show()

    def _refresh(self, folder_dir: str):
        self.lines.find_lines(folder_dir)
        self.GUI.refresh(self.lines)

    def _change_visibility(self, idx: int, flag: bool):
        self.lines.list[idx].visible = flag
        self.GUI.sc.plot(self.lines)

    def _change_color(self, idx: int, red: int, green: int, blue: int):
        self.lines.list[idx].color = [red, green, blue]
        self.GUI.sc.plot(self.lines)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ex = Software()
    sys.exit(app.exec_())
