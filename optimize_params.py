"""
# genetic algorithm notes
- look into simulated annealing
- make a "lean" version with:
	- no printing of position/activation @ every timestep, only print at end
	- dont print anything except the final position to stdout
"""

from typing import *
import subprocess
import copy
from math import dist

import json

if TYPE_CHECKING:
	from mypy_extensions import Arg
else:
	Arg = lambda t,s : t

from pyutil.util import (
	Path,mkdir,joinPath,
	strList_to_dict,ParamsDict,ModParamsDict,
	VecXY,dump_state,
	find_conn_idx,find_conn_idx_regex,
	genCmd_singlerun,
	dict_to_filename,
	keylist_access_nested_dict,
)

from pyutil.plot_pos import read_body_data,CoordsRotArr


def merge_params_with_mods(
		# base params
		params_base : ParamsDict,
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
	) -> ParamsDict:
	"""merges a params file with a special "mod" dict
	
	returns a modified copy of `params_base`, modified according to the contents of `params_mod`
	`params_base` is of the same form as a regular params.json file,
	but `params_mod` has the following structure:

	```python
	params_mod = {
			'__params__:Head.neurons.AWA.theta' : 2.0,
			'__params__:ChemoReceptors.alpha' : 2.0,
			'__conn__:Head,AWA,RIM,chem' : 10.0,
		}
	}
	```

	- keys starting with `__params__` map dot-separated keys to the nested params dict, to their desired values
	- keys starting with `__conn__` map comma-separated connection identifiers to their desired values
	
	### Parameters:
	 - `params_base : ParamsDict`
	   `params.json` style dict
	 - `params_mod : ModParamsDict` 
	   special dict to modify a copy of `params_base`
	
	### Returns:
	 - `ParamsDict` 
	   modified copy of `params_base`
	"""

	# copy the input dict
	output : dict = copy.deepcopy(params_base)

	# merge in the standard params
	for nested_keys,val in params_mod['params'].items():
		fin_dic,fin_key = keylist_access_nested_dict(
			d = output, 
			keys = nested_keys.split('.'),
		)
		fin_dic[fin_key] = val

	# merge in the connection modifiers
	for conn_key_str,conn_wgt in params_mod['conn'].items():
		conn_key = strList_to_dict(
			in_data = conn_key_str,
			keys_list = ['NS', 'from', 'to', 'type'],
			delim = ',',
		)

		# get the indecies of the connections whose weights need to be changed
		conn_idxs : List[Optional[int]] = find_conn_idx_regex(
			params_data = output, 
			conn_key = conn_key,
		)

		# set weights
		for cidx in conn_idxs:
			output[conn_key['NS']]['connections'][cidx]['weight'] = conn_wgt

	return output

ExtractorFunc = Callable[
	[
		Arg(Path, 'datadir'),
		Arg(ParamsDict, 'params'),
		Arg(bool, 'ret_nan'),
	], 
	Any, # return type
]

ExtractorReturnType = Any

def _extract_TEMPLATE(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> ExtractorReturnType:
	"""template function for extraction functions
	
	dont actually call this function. it contains documentation for the format of functions 
	`func_extract` taken by `evaluate_params()`
	
	### Parameters:
	 - `datadir : Path`   
	   output directory of data
	 - `params : ParamsDict`   
	   nested dictionary of params
	 - `ret_nan : bool`   
	   whether to return nan value (when process terminates in error)
	   (defaults to `False`)
	
	### Returns:
	 - `ExtractorReturnType` 
	   can return any data about the run
	
	### Raises:
	 - `NotImplementedError` : dont run this!
	"""
	
	raise NotImplementedError('this is a template function only!')


def _extract_finalpos(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> Tuple[float,float]:
	"""extract just the final head position
	
	### Returns:
	 - `Tuple[float,float]` 
	   head position
	"""
	if ret_nan:
		return ( float('nan'), float('nan') )	
	else: 
		bodydata : CoordsRotArr = read_body_data(datadir + 'body.dat')[-1,0]
		return ( bodydata['x'], bodydata['y'] )

def _extract_food_dist(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> float:
	"""extract euclidead distance from head to food
	
	### Returns:
	 - `float` 
	   dist from final head position to food
	"""
	if ret_nan:
		return float('nan')
	else:
		# get head pos
		bodydata : CoordsRotArr = read_body_data(datadir + 'body.dat')[-1,0]
		pos_head : Tuple[float,float] = ( bodydata['x'], bodydata['y'] )

		# get food pos
		pos_food : Tuple[float,float] = (
			params['ChemoReceptors']['foodPos']['x'],
			params['ChemoReceptors']['foodPos']['y'],
		)

		# return distance
		return dist(pos_head, pos_food)


def _extract_df_row(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> dict:
	# TODO: implement extracting more data, for parameter sweeps
	raise NotImplementedError('please implement me :(')


def evaluate_params(
		# base params
		params_base : ParamsDict,
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
		# root directory for run
		rootdir : Path = 'data/run/anneal/',
		coll : Path = 'input/objs_empty.tsv',
		# extract info from the final product
		func_extract : ExtractorFunc = _extract_food_dist,
		# command line args
		rand : Optional[bool] = None,
	) -> ExtractorReturnType:
	# TODO: document this
	
	# make dir
	outpath : str = f"{rootdir}{dict_to_filename(params_mod)}/"
	outpath_params : str = joinPath(outpath,'in-params.json')
	mkdir(outpath)

	# join params
	params_joined : dict = merge_params_with_mods(params_base, params_mod)

	# save modified params
	with open(outpath_params, 'w') as fout:
		json.dump(params_joined, fout, indent = '\t')

	# set up the command by passing kwargs down
	cmd : str = genCmd_singlerun(
		params = outpath_params,
		output = outpath,
		coll = coll,
		# **kwargs,
	)

	# run the process, write stderr and stdout to the log file
	with open(outpath + 'log.txt', 'w') as f_log:
		p = subprocess.Popen(
			cmd, 
			stderr = subprocess.STDOUT,
			stdout = f_log,
		)

	# wait for command to finish
	p.wait()

		
	if p.returncode:
		print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n        {str(p.args)}')


	return func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(p.returncode),
	)




def gen_random_state(
		ModParamsDict
	)
