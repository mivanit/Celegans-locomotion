"""runs `singlerun` with food on left, right, and no food"""

import os
from typing import *
import subprocess
import json
from copy import deepcopy

import numpy as np
from nptyping import NDArray

from pydbg import dbg

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
			foodPos : Optional[str] = None,
			**kwargs,
		):
		"""runs multiple trials of the simulation with food on left, right, and absent
		
		runs each of the following:
		```python
		dct_runs : Dict[str,str] = {
			'food_none/' : 'DISABLE',
			'food_left/' : f'{-food_x},{food_y}',
			'food_right/' : f'{food_x},{food_y}',
		}
		```
		with `food_x`, `food_y` extracted from `foodPos` parameter, or `params` json file if `foodPos is None`
		
		### Parameters:
		 - `output : Path`   
		   output path, will create folders for each food position inside this directory
		   (defaults to `'data/run/'`)
		 - `foodPos : Optional[str]`   
		   food position tuple
		   (defaults to `None`)
		
		### Raises:
		 - `TypeError` : if `foodPos` cant be read
		 - `KeyError` : shouldn't ever be raised -- state *should* be inacessible
		"""

		# get food position
		if foodPos is None:
			# from params json
			with open(kwargs['params'], 'r') as fin_json:
				params_json : dict = json.load(fin_json)

				food_x = params_json["ChemoReceptors"]["foodPos"]["x"]
				food_y = params_json["ChemoReceptors"]["foodPos"]["y"]
		else:
			# or from CLI (takes priority, if given)
			if isinstance(foodPos, str):
				food_x,food_y = foodPos.split(',')
			elif isinstance(foodPos, tuple):
				food_x,food_y = foodPos
			else:
				raise TypeError(f'couldnt read foodpos, expected str or tuple:   {foodPos}')
			food_x = float(food_x)
			food_y = float(food_y)

		# take absolute value for left/right to match
		food_x = abs(food_x)

		# make sure we dont pass the food pos further down
		if 'foodPos' in kwargs:
			raise KeyError(f'"foodPos" still specified? this should be inacessible')

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
			special_scaling_map : Optional[Dict[str,float]] = None,
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
		if conn_key['to'].endswith('*'):
			# if wildcard given, find every connection that matches
			conn_idxs : List[int] = list()
			
			conn_key_temp : dict = deepcopy(conn_key)
			
			for nrn in params_data[conn_key['NS']]['neurons']:
				# loop over neuron names, check if they match
				# REVIEW: this isnt full regex, but whatever
				dbg(nrn)
				if nrn.startswith(conn_key['to'].split('*')[0]):
					conn_key_temp['to'] = nrn
					dbg(conn_key_temp)
					cidx_temp : Optional[int] = find_conn_idx(
						params_data[conn_key_temp['NS']]['connections'],
						conn_key_temp,
					)
					dbg(cidx_temp)
					# append to list, but only if an existing connection is found
					# note that this behavior differs from when no wildcard is given,
					# in that new connections will not be created
					if cidx_temp is not None:
						conn_idxs.append(cidx_temp)
		else:
			if special_scaling_map is not None:
				raise ValueError(f"`special_scaling_map` specified, but no wildcard given in neuron name:   {special_scaling_map}    {conn_key['to']}")

			# if no wildcard specified, just get the one connection
			conn_idxs : List[int] = [ find_conn_idx(
				params_data[conn_key['NS']]['connections'],
				conn_key,
			) ]

		if None in conn_idxs:
			# if the connection doesnt exist, add it
			params_data[conn_key['NS']]['connections'].append({
				'from' : conn_key['from'],
				'to' : conn_key['to'],
				'type' : conn_key['type'],
				'weight' : float('nan'),
			})
		
			# if the connection still doesn't exist, something has gone wrong
			conn_idxs = [ find_conn_idx(
				params_data[conn_key['NS']]['connections'],
				conn_key,
			) ]

		if (None in conn_idxs) or (len(conn_idxs) == 0):
			raise KeyError(f'couldnt find connection index -- this state should be innaccessible.   list:  {conn_idxs}')


		# figure out the range of values to try
		weight_vals : NDArray = SPACE_GENERATOR_MAPPING[conn_range['scale']](
			conn_range['min'], 
			conn_range['max'], 
			conn_range['npts'],
		)
		
		count : int = 1
		count_max : int = len(weight_vals)

		print('> will modify connections:')
		for cidx in conn_idxs:
			print('\t>>  ' + str(params_data[conn_key['NS']]['connections'][cidx]))
		print('> will try weights:')
		print(f'\t>>  {weight_vals}')
		input('press enter to continue...')

		# set up for scaling the weight
		wgt_scale : float = 1.0
		if special_scaling_map is None:
			special_scaling_map = dict()
		
		# run for each value of connection strength
		for wgt in weight_vals:
			print(f'> running for weight {wgt} \t ({count} / {count_max})')
			# make dir
			outpath : str = f"{output}{conn_key['from']}-{conn_key['to'].replace('*','x')}_{wgt:.5}/"
			outpath_params : str = joinPath(outpath,'params.json')
			mkdir(outpath)

			# set weights
			for cidx in conn_idxs:
				# scale the weight if the neuron name is in the map
				cidx_nrn_to : str = params_data[conn_key['NS']]['connections'][cidx]['to']
				if cidx_nrn_to in special_scaling_map:
					wgt_scale = special_scaling_map[cidx_nrn_to]
				else:
					wgt_scale = 1.0

				# set the new weight
				params_data[conn_key['NS']]['connections'][cidx]['weight'] = wgt * wgt_scale

			# save modified params
			with open(outpath_params, 'w') as fout:
				json.dump(params_data, fout, indent = '\t')

			# run
			Launchers.multi_food_run(
				output = outpath,
				params = outpath_params,
				**kwargs
			)

			count += 1

	@staticmethod
	def sweep_hardcoded_turning_RMDx(
			output : Path = 'data/run/',
			conn_range : Union[dict,tuple,str] = '0.0,1.0,lin,3',
			nrn_from : str = 'CONST',
			params : Path = 'input/params.json',
			scaling_map_sign_dorsal : float = 1.0,
			**kwargs,
		):

		Launchers.sweep_conn_weight(
			output = output,
			conn_key = ('Head', nrn_from,'RMD*','chem'),
			conn_range = conn_range,
			params = params,
			special_scaling_map = {
				'RMDD' : scaling_map_sign_dorsal,
				'RMDV' : - scaling_map_sign_dorsal,
			},
			**kwargs,
		)
	



if __name__ == "__main__":
	import fire
	fire.Fire(Launchers)