import os, json, subprocess
from typing import List, Dict, Tuple, Union
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QGroupBox, QDialog, QPushButton


from software.Utils.util import mkdir, dump_state, genCmd_singlerun
from software.Utils.FILE_DEFAULT_NAME import PARAMS_FILE_NAME
from software.Signal.Signal import MYSIGNAL

class Launcher(QDialog):
    def __init__(self, rootdir, parent=None):
        super(Launcher, self).__init__()
        self.rootdir = rootdir
        self.parent = parent
        dump_state(locals(), rootdir)
        cmd: str = genCmd_singlerun(
            output=rootdir+"/",
            params=rootdir+"/"+PARAMS_FILE_NAME
        )
        print(cmd)
        # run the process, write stderr and stdout to the log file
        with open(rootdir + '/log.txt', 'w') as f_log:
            p = subprocess.Popen(
                cmd,
                stderr=subprocess.STDOUT,
                stdout=f_log,
            )


if __name__ == '__main__':
    Launcher(os.getcwd()+"/test/")



