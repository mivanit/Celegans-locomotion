"""runs `singlerun` with food on left, right, and no food"""

import os
from typing import *
import subprocess
import json

from pyutil.util import Path,genCmd_singlerun,strList_to_dict


def multi_food_run(
		output : Path = 'data/run/',
		food_x : float = None,
		food_y : float = None,
		**kwargs,
	):

	if (food_x is None) or (food_y is None):
		with open(kwargs['params'], 'r') as fin_json:
			params_json : dict = json.load(fin_json)

			if food_x is None:
				food_x = params_json["ChemoReceptors"]["foodPos"]["x"]

			if food_y is None:
				food_y = params_json["ChemoReceptors"]["foodPos"]["y"]

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




def test_conn_range(
		output : Path = 'data/run/',
		conn_key : Union[dict,str] = 'Head,AWA,RIM,chem',
		conn_range : Union[dict,str] = '0.0,0.1,log',
		food_x : float = None,
		food_y : float = None,
		**kwargs,
	):

	# convert input string-lists to dictionaries 
	# (useful as shorthand when using python-fire CLI)
	conn_key = strList_to_dict(
		in_data = conn_key,
		keys_list = ['NS', 'from', 'to', 'type'],
		type_map = {'from' : float, 'to' : float},
	)

	conn_range = strList_to_dict(
		in_data = conn_range,
		keys_list = ['min', 'max', 'scale'],
		type_map = {'min' : float, 'max' : float},
	)


	# find the appropriate connection to modify
	





if __name__ == "__main__":
	import fire
	fire.Fire(multi_food_run)