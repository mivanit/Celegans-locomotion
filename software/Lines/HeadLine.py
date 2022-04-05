import os, json
import random

from software.Utils.FILE_DEFAULT_NAME import BODY_FILE_NAME, COLLOBJS_FILE_NAME, PARAMS_FILE_NAME
from software.Utils.read_runs import read_body_data
from software.Lines.collision_object import read_collobjs_tsv

class HeadLine:
    def __init__(self, folder_dir:str):
        self.folder_dir = folder_dir
        self.data = read_body_data(folder_dir + "/" + BODY_FILE_NAME)
        self.collobjs = None
        self.params = None
        self.visible = True
        self.color = [int(random.random()*255) for i in range(3)]
        self._initialize_collobjs()
        self._initialize_para()

    def _initialize_collobjs(self):
        coll_dir = self.folder_dir + "/" + COLLOBJS_FILE_NAME
        if os.path.isfile(coll_dir):
            self.collobjs = read_collobjs_tsv(coll_dir)
        else:
            print(f'  >> WARNING: could not find file, skipping: {coll_dir}')

    def _initialize_para(self):
        params = self.folder_dir + "/" + PARAMS_FILE_NAME
        with open(params, 'r') as fin:
            self.params: dict = json.load(fin)




