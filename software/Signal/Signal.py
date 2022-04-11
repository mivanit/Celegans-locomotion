import sys
import PyQt5.QtWidgets as PQW
import PyQt5.QtCore as PQC


class CustSignal(PQC.QObject):
    # 无参数的信号
    Open_Folder = PQC.pyqtSignal(str)
    Color_Change = PQC.pyqtSignal([int, int, int, int])  # the first num for idx, the rest three for rgb
    Visibility_Change = PQC.pyqtSignal([int, bool])
    Open_Manger = PQC.pyqtSignal(str)
    Change_param = PQC.pyqtSignal(list)
    Create_line = PQC.pyqtSignal()

    def __init__(self, parent=None):
        super(CustSignal, self).__init__(parent)


MYSIGNAL = CustSignal()
