"""runs `sim` with food on left, right, and no food"""

import os
from typing import *
import subprocess
import json
from copy import deepcopy

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore

# from pydbg import dbg # type: ignore

from pyutil.util import (
	Path,mkdir,joinPath,dump_state,
	strList_to_dict,
	genCmd_singlerun,
	keylist_access_nested_dict,
)

from pyutil.eval_run import find_conn_idx,find_conn_idx_regex


SPACE_GENERATOR_MAPPING : Dict[str,Callable] = {
	'lin' : np.linspace,
	'log' : np.logspace,
}


class Launchers(object):
	@staticmethod
	def multi_food_run(
			rootdir : Path = 'data/run/',
			foodPos : Union[None,str,Tuple[float,float]] = (-0.003,0.005),
			angle : Optional[float] = 1.57,
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
		 - `rootdir : Path`   
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
		mkdir(rootdir)

		# save state
		dump_state(locals(), rootdir)
		
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
			out_path : str = joinPath(rootdir,name)
			
			mkdir(out_path)

			# set up the command by passing kwargs down
			cmd : List[str] = genCmd_singlerun(
				output = out_path,
				foodPos = foodPos,
				angle = angle,
				**kwargs,
			).split(' ')

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
				print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n\t{str(p.args)}')	
			else:
				print(f'  >>  process complete: {name}')



	@staticmethod
	def sweep_conn_weight(
			rootdir : Path = 'data/run/',
			conn_key : Union[dict,tuple,str] = 'Head,AWA,RIM,chem',
			conn_range : Union[dict,tuple,str] = '0.0,1.0,lin,3',
			params : Path = 'input/params.json',
			special_scaling_map : Optional[Dict[str,float]] = None,
			ASK_CONTINUE : bool = True,
			**kwargs,
		):

		# create output dir
		mkdir(rootdir)

		# save state
		dump_state(locals(), rootdir)

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
		conn_idxs : List[Optional[int]] = find_conn_idx_regex(
			params_data = params_data,
			conn_key = conn_key,
			# special_scaling_map = special_scaling_map,
		)

		if None in conn_idxs:
			# REVIEW: this is probably not good behavior
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

		if ASK_CONTINUE:
			input('press enter to continue...')

		# set up for scaling the weight
		wgt_scale : float = 1.0
		if special_scaling_map is None:
			special_scaling_map = dict()
		
		# run for each value of connection strength
		for wgt in weight_vals:
			print(f'> running for weight {wgt} \t ({count} / {count_max})')
			# make dir
			outpath : str = f"{rootdir}{conn_key['from']}-{conn_key['to'].replace('*','x')}_{wgt:.5}/"
			outpath_params : str = joinPath(outpath,'in-params.json')
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
				rootdir = outpath,
				params = outpath_params,
				**kwargs
			)

			count += 1

	@staticmethod
	def sweep_hardcoded_turning_RMDx(
			rootdir : Path = 'data/run/',
			conn_range : Union[dict,tuple,str] = '0.0,1.0,lin,3',
			nrn_from : str = 'CONST',
			params : Path = 'input/params.json',
			scaling_map_sign_dorsal : float = 1.0,
			**kwargs,
		):

		Launchers.sweep_conn_weight(
			rootdir = rootdir,
			conn_key = ('Head', nrn_from,'RMD*','chem'),
			conn_range = conn_range,
			params = params,
			special_scaling_map = {
				'RMDD' : scaling_map_sign_dorsal,
				'RMDV' : - scaling_map_sign_dorsal,
			},
			**kwargs,
		)

	@staticmethod
	def sweep_param(
			rootdir : Path = 'data/run/',
			param_key : Union[tuple,str] = 'ChemoReceptors.alpha',
			param_range : Union[dict,tuple,str] = '0.0,1.0,lin,3',
			params : Path = 'input/params.json',
			multi_food : bool = False,
			ASK_CONTINUE : bool = True,
			**kwargs,
		):

		# create output dir
		mkdir(rootdir)

		# save state
		dump_state(locals(), rootdir)

		# open base json
		with open(params, 'r') as fin_json:
			params_data : dict = json.load(fin_json)

		# convert input string-lists
		# (useful as shorthand when using python-fire CLI)

		# split up path to parameter by dot
		param_key_tup : Tuple[str,...] = (
			tuple(param_key.split('.'))
			if isinstance(param_key, str)
			else tuple(param_key)
		)
		param_key_sdot : str = '.'.join(param_key_tup)

		# convert into a dict
		param_range_dict : Dict[str,Any] = strList_to_dict(
			in_data = param_range,
			keys_list = ['min', 'max', 'scale', 'npts'],
			type_map = {'min' : float, 'max' : float, 'npts' : int},
		)

		print(f'>> parameter to modify: {param_key_sdot}')
		print(f'>> range of values: {param_range_dict}')
	
		param_fin_dict : dict = params_data
		param_fin_key : str = ''
		try:
			param_fin_dict,param_fin_key = keylist_access_nested_dict(params_data, param_key_tup)
		except KeyError as ex:
			print(f'\n{param_key_sdot} was not a valid parameter for the params file read from {params}. Be sure that the parameter you want to modify exists in the json file.\n')
			raise ex
			exit(1)

		# figure out the range of values to try
		param_vals : NDArray = SPACE_GENERATOR_MAPPING[param_range_dict['scale']](
			param_range_dict['min'], 
			param_range_dict['max'], 
			param_range_dict['npts'],
		)
		
		count : int = 1
		count_max : int = len(param_vals)

		print(f'> will modify parameter: {param_key_sdot}\n\t>>  {param_fin_dict}\t-->\t{param_fin_key}')
		print(f'> will try {len(param_vals)} values:\n\t>>  {param_vals}')
		if ASK_CONTINUE:
			input('press enter to continue...')
		
		# run for each value of connection strength
		for pv in param_vals:
			print(f'> running for param val {pv} \t ({count} / {count_max})')

			# make dir
			outpath : str = f"{rootdir}{param_key_sdot}_{pv:.5}/"
			outpath_params : str = joinPath(outpath,'in-params.json')
			mkdir(outpath)

			# set value
			param_fin_dict[param_fin_key] = pv

			# save modified params
			with open(outpath_params, 'w') as fout:
				json.dump(params_data, fout, indent = '\t')

			# run
			if multi_food:
				Launchers.multi_food_run(
					rootdir = outpath,
					params = outpath_params,
					**kwargs
				)
			else:
				cmd : str = genCmd_singlerun(
					output = outpath,
					params = outpath_params,
					**kwargs,
				)

				print(cmd)

				# run the process, write stderr and stdout to the log file
				with open(outpath + 'log.txt', 'w') as f_log:
					p = subprocess.Popen(
						cmd, 
						stderr = subprocess.STDOUT,
						stdout = f_log,
					)
			
			count += 1
	



if __name__ == "__main__":
	import fire # type: ignore
	fire.Fire(Launchers)

