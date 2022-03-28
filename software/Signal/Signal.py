import sys
import PyQt5.QtWidgets as PQW
import PyQt5.QtCore as PQC

class MyWidget(PQW.QWidget):
    # 无参数的信号
    Signal_NoParameters = PQC.pyqtSignal()
    # 带一个参数(整数)的信号
    Signal_OneParameter = PQC.pyqtSignal(int)
    # 带一个参数(整数或者字符串)的重载版本的信号
    Signal_OneParameter_Overload = PQC.pyqtSignal([int],[str])
    # 带两个参数(整数,字符串)的信号
    Signal_TwoParameters = PQC.pyqtSignal(int,str)
    # 带两个参数([整数,整数]或者[整数,字符串])的重载版本的信号
    Signal_TwoParameters_Overload = PQC.pyqtSignal([int,int],[int,str])