"""runs `singlerun` with food on left, right, and no food"""

import os
from typing import *
import subprocess

from pyutil.util import Path,genCmd_singlerun


def multi_food_run(
		output : Path = 'data/run/',
		food_x : float = 0.005,
		food_y : float = 0.005,
		**kwargs,
	):

	if not os.path.isdir(output):
		os.mkdir(output)

	food_x = abs(food_x)

	if 'foodPos' in kwargs:
		raise KeyError(f'"foodPos" specified in `multi_food_run`, which is not allowed')


	dct_runs : Dict[str,str] = {
		'food_none/' : 'DISABLE',
		'food_left/' : f'{-food_x},{food_y}',
		'food_right/' : f'{food_x},{food_y}',
	}

	dct_procs : dict = dict()

	for name,foodPos in dct_runs.items():

		out_path : str = output + name
		
		if not os.path.isdir(out_path):
			os.mkdir(out_path)

		cmd : str = genCmd_singlerun(
			output = out_path,
			foodPos = foodPos,
			**kwargs,
		)

		print(cmd)

		with open(out_path + 'log.txt', 'w') as f_log:
			p = subprocess.Popen(
				cmd, 
				stderr = subprocess.STDOUT,
				stdout = f_log,
			)

		dct_procs[name] = p

	for name,p in dct_procs.items():
		p.wait()
		
		if p.returncode:
			print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n\t{p.args}')	
		else:
			print(f'  >>  process complete: {name}')



if __name__ == "__main__":
	import fire
	fire.Fire(multi_food_run)