from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem
from PyQt5.QtGui import QColor

from software.Lines.LineSet import LineSet

class LineTable(QTableWidget):
    def __init__(self):
        super(LineTable, self).__init__()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['color', 'dir'])

    def refresh(self, line_list: LineSet):
        self.setRowCount(len(line_list.list))
        for idx, line in enumerate(line_list.list):
            new_color = QTableWidgetItem(" ")
            new_color.setBackground(QColor(int(line.color[0]*255),int(line.color[1]*255), int(line.color[2]*255)))
            self.setItem(idx, 0, new_color)
            new_item = QTableWidgetItem(line.folder_dir[23:])
            self.setItem(idx, 1, new_item)



