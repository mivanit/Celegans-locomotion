
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QColorDialog
from PyQt5.QtGui import QColor

from software.Lines.LineSet import LineSet
from software.Signal.Signal import MYSIGNAL

class LineTable(QTableWidget):
    def __init__(self):
        super(LineTable, self).__init__()
        self.setColumnCount(3)
        # self.setHorizontalHeaderLabels(['show', 'color', 'dir'])
        self.verticalHeader().setVisible(True)
        self.horizontalHeader().setVisible(False)
        self.setColumnWidth(0, 30)
        self.setColumnWidth(1, 30)
        self.setColumnWidth(2, 250)
        self.cellDoubleClicked.connect(self.change_details)
        self.cellChanged.connect(self.update_info)
        self.index_end = 0

    def refresh(self, line_list: LineSet):
        self.setRowCount(len(line_list.list))
        idx = 0
        for idx in range(self.index_end, len(line_list.list)):
            line = line_list.list[idx]
            chkBoxItem = QTableWidgetItem()
            chkBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            if line.visible:
                chkBoxItem.setCheckState(Qt.Checked)
            else:
                chkBoxItem.setCheckState(Qt.Unchecked)
            # chkBoxItem.checkState()
            self.setItem(idx, 0, chkBoxItem)
            new_color = QTableWidgetItem()
            new_color.setFlags(Qt.ItemIsEnabled)
            new_color.setBackground(QColor(line.color[0], line.color[1], line.color[2]))
            self.setItem(idx, 1, new_color)
            new_item = QTableWidgetItem(line.folder_dir)
            new_item.setFlags(Qt.ItemIsEnabled)
            new_item.setTextAlignment(Qt.AlignRight)
            self.setItem(idx, 2, new_item)
            self.setRowHeight(idx, 25)
        self.index_end = idx

    def change_details(self, row, col):
        if col == 1:
            col = QColorDialog.getColor()
            if col.isValid():
                self.item(row, 1).setBackground(col.toRgb())
        if col == 2:
            MYSIGNAL.Open_Manger.emit(self.item(row, 2).text())

    def update_info(self, row, col):
        if col == 0:
            flag = True if self.item(row, col).checkState() == Qt.Checked else False
            MYSIGNAL.Visibility_Change.emit(row, flag)
        if col == 1:
            color = self.item(row, col).background().color().toRgb()
            MYSIGNAL.Color_Change.emit(row, color.red(), color.green(), color.blue())









