import os
from typing import List
from software.Lines.HeadLine import HeadLine
from software.Utils.FILE_DEFAULT_NAME import BODY_FILE_NAME


class LineSet:
    def __init__(self):
        self.list: List[HeadLine] = []

    def find_lines(self, start_dir):
        dir_res = os.listdir(start_dir)
        for path in dir_res:
            temp_path = start_dir + '/' + path
            if os.path.isfile(temp_path) and path == BODY_FILE_NAME:
                self.list.append(HeadLine(start_dir))
            if os.path.isdir(temp_path):
                self.find_lines(temp_path)
        print("search finish, number is", len(self.list))
