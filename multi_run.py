"""runs `singlerun` with food on left, right, and no food"""

import os
from typing import *
import subprocess
import json

import numpy as np
from nptyping import NDArray

from pyutil.util import (
	Path,mkdir,joinPath,dump_state,
	strList_to_dict,find_conn_idx,
	genCmd_singlerun,
)


SPACE_GENERATOR_MAPPING : Dict[str,Callable] = {
	'lin' : np.linspace,
	'log' : np.logspace,
}


class Launchers(object):
	@staticmethod
	def multi_food_run(
			output : Path = 'data/run/',
			foodPos : str = None,
			**kwargs,
		):

		# get food position
		if foodPos is None:
			# from params json
			with open(kwargs['params'], 'r') as fin_json:
				params_json : dict = json.load(fin_json)

				food_x = params_json["ChemoReceptors"]["foodPos"]["x"]
				food_y = params_json["ChemoReceptors"]["foodPos"]["y"]
		else:
			# or from CLI (takes priority, if given)
			food_x,food_y = foodPos.split(',')
			food_x = float(food_x)
			food_y = float(food_y)

		# take absolute value for left/right to match
		food_x = abs(food_x)

		# make sure we dont pass the food pos further down
		if 'foodPos' in kwargs:
			raise KeyError(f'"foodPos" still specified? this should be innacessible')

		# create output dir
		mkdir(output)

		# save state
		dump_state(locals(), output)
		
		# set up the different runs
		dct_runs : Dict[str,str] = {
			'food_none/' : 'DISABLE',
			'food_left/' : f'{-food_x},{food_y}',
			'food_right/' : f'{food_x},{food_y}',
		}

		# dictionary of running processes
		dct_procs : dict = dict()

		# start each process
		for name,foodPos in dct_runs.items():

			# make the output dir
			out_path : str = output + name
			
			mkdir(out_path)

			# set up the command by passing kwargs down
			cmd : str = genCmd_singlerun(
				output = out_path,
				foodPos = foodPos,
				**kwargs,
			)

			print(cmd)

			# run the process, write stderr and stdout to the log file
			with open(out_path + 'log.txt', 'w') as f_log:
				p = subprocess.Popen(
					cmd, 
					stderr = subprocess.STDOUT,
					stdout = f_log,
				)

			# store process in dict for later
			dct_procs[name] = p


		# wait for all of them to finish
		for name,p in dct_procs.items():
			p.wait()
			
			if p.returncode:
				print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n\t{p.args}')	
			else:
				print(f'  >>  process complete: {name}')



	@staticmethod
	def sweep_conn_weight(
			output : Path = 'data/run/',
			conn_key : Union[dict,tuple,str] = 'Head,AWA,RIM,chem',
			conn_range : Union[dict,tuple,str] = '0.0,1.0,lin,3',
			params : Path = 'input/params.json',
			**kwargs,
		):

		# create output dir
		mkdir(output)

		# save state
		dump_state(locals(), output)

		# open base json
		with open(params, 'r') as fin_json:
			params_data : dict = json.load(fin_json)

		# convert input string-lists to dictionaries 
		# (useful as shorthand when using python-fire CLI)
		conn_key = strList_to_dict(
			in_data = conn_key,
			keys_list = ['NS', 'from', 'to', 'type'],
		)

		conn_range = strList_to_dict(
			in_data = conn_range,
			keys_list = ['min', 'max', 'scale', 'npts'],
			type_map = {'min' : float, 'max' : float, 'npts' : int},
		)

		print(f'>> connection to modify: {conn_key}')
		print(f'>> range of values: {conn_range}')


		# find the appropriate connection to modify
		conn_idx : int = find_conn_idx(
			params_data[conn_key['NS']]['connections'],
			conn_key,
		)

		if conn_idx is None:
			# if the connection doesnt exist, add it
			params_data[conn_key['NS']]['connections'].append({
				'from' : conn_key['from'],
				'to' : conn_key['to'],
				'type' : conn_key['type'],
				'weight' : float('nan'),
			})
		
		# if the connection still doesn't exist, something has gone wrong
		conn_idx = find_conn_idx(
			params_data[conn_key['NS']]['connections'],
			conn_key,
		)

		if conn_idx is None:
			raise KeyError('couldnt find connection index -- this state should be innaccessible')


		# figure out the range of values to try
		weight_vals : NDArray = SPACE_GENERATOR_MAPPING[conn_range['scale']](
			conn_range['min'], 
			conn_range['max'], 
			conn_range['npts'],
		)
		

		# run for each value of connection strength
		for wgt in weight_vals:
			print(f'> running for weight {wgt}')
			# make dir
			outpath : str = f"{output}{conn_key['from']}-{conn_key['to']}_{wgt:.4}/"
			outpath_params : str = joinPath(outpath,'params.json')
			mkdir(outpath)

			# set weight
			params_data[conn_key['NS']]['connections'][conn_idx]['weight'] = wgt

			# save modified params
			with open(outpath_params, 'w') as fout:
				json.dump(params_data, fout, indent = '\t')

			# run
			Launchers.multi_food_run(
				output = outpath,
				params = outpath_params,
				**kwargs
			)
	



if __name__ == "__main__":
	import fire
	fire.Fire(Launchers)