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
import random

import json

if TYPE_CHECKING:
	from mypy_extensions import Arg
else:
	Arg = lambda t,s : t

from pyutil.util import (
	ModTypes, Path,mkdir,joinPath,
	strList_to_dict,ParamsDict,ModParamsDict,ModParamsRanges,
	VecXY,dump_state,
	find_conn_idx,find_conn_idx_regex,
	genCmd_singlerun,
	dict_to_filename,
	keylist_access_nested_dict,
)

from pyutil.plot_pos import read_body_data,CoordsRotArr


"""

 #    # ###### #####   ####  ######
 ##  ## #      #    # #    # #
 # ## # #####  #    # #      #####
 #    # #      #####  #  ### #
 #    # #      #   #  #    # #
 #    # ###### #    #  ####  ######

"""

def merge_params_with_mods(
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
		# base params
		params_base : ParamsDict,
	) -> ParamsDict:
	"""merges a params file with a special "mod" dict
	
	returns a modified copy of `params_base`, modified according to the contents of `params_mod`
	`params_base` is of the same form as a regular params.json file,
	but `params_mod` has the following structure:

	```python
	params_mod = {
			('params','Head.neurons.AWA.theta' : 2.0,
			('params','ChemoReceptors.alpha' : 2.0,
			('conn','Head,AWA,RIM,chem' : 10.0,
		}
	}
	```

	- keys starting with `params` map dot-separated keys to the nested params dict, to their desired values
	- keys starting with `conn` map comma-separated connection identifiers to their desired values
	
	### Parameters:
	 - `params_mod : ModParamsDict` 
	   special dict to modify a copy of `params_base`
	 - `params_base : ParamsDict`
	   `params.json` style dict
	
	### Returns:
	 - `ParamsDict` 
	   modified copy of `params_base`
	"""

	# copy the input dict
	output : dict = copy.deepcopy(params_base)

	# REVIEW: why did i even refactor this when im making everything editable through params json anyway?
	for tup_key,val in params_mod.items():
		# merge in the standard params
		if tup_key.mod_type == ModTypes.params:
			
			nested_keys : str = tup_key.path

			fin_dic,fin_key = keylist_access_nested_dict(
				d = output, 
				keys = nested_keys.split('.'),
			)
			fin_dic[fin_key] = val

		elif tup_key.mod_type == ModTypes.conn:
			# merge in the connection modifiers
			conn_key_str : str = tup_key.path
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
				output[conn_key['NS']]['connections'][cidx]['weight'] = val
		else:
			raise NotImplementedError(f'given key type {tup_key.mod_type} unknown')

	return output


"""

 ###### #    # ##### #####    ##    ####  #####
 #       #  #    #   #    #  #  #  #    #   #
 #####    ##     #   #    # #    # #        #
 #        ##     #   #####  ###### #        #
 #       #  #    #   #   #  #    # #    #   #
 ###### #    #   #   #    # #    #  ####    #

"""

ExtractorFunc = Callable[
	[
		Arg(Path, 'datadir'),
		Arg(ParamsDict, 'params'),
		Arg(bool, 'ret_nan'),
	], 
	Any, # return type
]

ExtractorReturnType = Any

def _wrapper_extract(
		proc, 
		func_extract : ExtractorFunc, 
		outpath : Path, 
		params_joined : ParamsDict,
	):	
	# wait for command to finish
	proc.wait()
		
	if proc.returncode:
		print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n        {str(proc.args)}')

	return func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(proc.returncode),
	)



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
		pos_head : VecXY = VecXY( bodydata['x'], bodydata['y'] )

		# get food pos
		pos_food : VecXY = VecXY(
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


"""

 ###### #    #   ##   #
 #      #    #  #  #  #
 #####  #    # #    # #
 #      #    # ###### #
 #       #  #  #    # #
 ######   ##   #    # ######

"""

def setup_evaluate_params(
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
		# base params
		params_base : ParamsDict,
		# root directory for run
		rootdir : Path = 'data/run/anneal/',
		# extract info from the final product
		func_extract : ExtractorFunc = _extract_food_dist,
		# command line args
		rand : Optional[bool] = None,
	) -> ExtractorReturnType:
	# TODO: document this
	
	# make dir
	outpath : Path = f"{rootdir}{dict_to_filename(params_mod)}/"
	outpath_params : Path = joinPath(outpath,'in-params.json')
	mkdir(outpath)

	# join params
	params_joined : ParamsDict = merge_params_with_mods(params_base, params_mod)

	# modify CLI parameters from mod
	merge_params_with_mods(params_base, params_mod)

	# save modified params
	with open(outpath_params, 'w') as fout:
		json.dump(params_joined, fout, indent = '\t')

	# set up the command by passing kwargs down
	cmd : str = genCmd_singlerun(
		params = outpath_params,
		output = outpath,
		# **kwargs,
	)

	# run the process, write stderr and stdout to the log file
	with open(outpath + 'log.txt', 'w') as f_log:
		proc = subprocess.Popen(
			cmd, 
			stderr = subprocess.STDOUT,
			stdout = f_log,
		)

	return (proc, outpath, params_joined)
	
def evaluate_params(
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
		# base params
		params_base : ParamsDict,
		# root directory for run
		rootdir : Path = 'data/run/anneal/',
		# extract info from the final product
		func_extract : ExtractorFunc = _extract_food_dist,
		# command line args
		rand : Optional[bool] = None,
	) -> ExtractorReturnType:
	
	proc, outpath, params_joined = setup_evaluate_params(
		params_mod = params_mod,
		params_base = params_base,
		rootdir= rootdir,
	)

	# wait for command to finish
	proc.wait()
		
	if proc.returncode:
		print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n        {str(proc.args)}')

	return func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(proc.returncode),
	)


"""

  ####  ###### #    # ######
 #    # #      ##   # #
 #      #####  # #  # #####
 #  ### #      #  # # #
 #    # #      #   ## #
  ####  ###### #    # ######

"""

def mutate_state(
		params_mod : ModParamsDict,
		ranges : ModParamsRanges,
		sigma : float = 0.1,
	) -> None:
	
	# choose a variable to mutate
	choice_key : str = random.choice(list(params_mod.keys()))
	choice_val : float = params_mod[choice_key]

	# modify the value according to the range
	delta_val : float = random.gauss(0, sigma)
	params_mod[choice_key] = choice_val + delta_val



def combine_genotypes(
		pmod_A : ModParamsDict,
		pmod_B : ModParamsDict,
		noise_sigma : float = 0.1,
		threshold_noise : float = 0.00001,
	) -> ModParamsDict:
	"""combines `pmod_A, pmod_B` into a single genotype
	
	when modifying an individual component of the genotypes,
	if the difference is over `threshold_noise`,
	then we take the average between the values 
	and add noise using `noise_sigma` as sigma for normal distribution
	if difference < `threshold_noise`, we just take the average
	
	notes:
	- keys of `pmod_A, pmod_B` should match
	- order of `pmod_A, pmod_B` shouldnt matter
	
	### Parameters:
	 - `pmod_A : ModParamsDict`   
	   [description]
	 - `pmod_B : ModParamsDict`   
	   [description]
	 - `ranges : ModParamsRanges`   
	   [description]
	 - `noise_sigma : float`   
	   [description]
	
	### Returns:
	 - `ModParamsDict` 
	   [description]
	"""

	pmod_out : ModParamsDict = dict()

	# assert that keys match
	assert all(
		(k in pmod_B) 
		for k in pmod_A.keys()
	), 'keys dont match!'

	for key in pmod_A.keys():
		# set new val to average
		val : float = (pmod_A[key] + pmod_B[key]) / 2.0
		
		# add noise if difference is big enough
		val_range : float = abs(pmod_A[key] - pmod_B[key])
		if val_range > threshold_noise:
			val += random.gauss(0.0, val_range * noise_sigma)

		# store new val
		pmod_out[key] = val

	return pmod_out

















