from collections import defaultdict
from typing import *
import subprocess
import copy,sys
import os
from math import dist,isnan
import random
import json

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore
from pydbg import dbg # type: ignore

if TYPE_CHECKING:
	from mypy_extensions import Arg
else:
	Arg = lambda t,s : t

__EXPECTED_PATH__ : str = 'pyutil.eval_run'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import *
from pyutil.params import *
from pyutil.read_runs import read_body_data


"""

 ##### #   # #####  # #    #  ####
   #    # #  #    # # ##   # #    #
   #     #   #    # # # #  # #
   #     #   #####  # #  # # #  ###
   #     #   #      # #   ## #    #
   #     #   #      # #    #  ####

"""

ExtractorReturnType = Any

ExtractorFunc = Callable[
	[
		Arg(Path, 'datadir'),
		Arg(ParamsDict, 'params'),
		Arg(bool, 'ret_nan'),
	], 
	ExtractorReturnType, # return type
]

MultiExtractorFunc = Callable[
	[
		Arg(Path, 'datadir'),
		Arg(ParamsDict, 'params'),
		Arg(bool, 'ret_nan'),
	], 
	float, # return type
]



"""

 #    # #####    ##   #####
 #    # #    #  #  #  #    #
 #    # #    # #    # #    #
 # ## # #####  ###### #####
 ##  ## #   #  #    # #
 #    # #    # #    # #

"""

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

	result : ExtractorReturnType = func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(proc.returncode),
	)

	with open(joinPath(outpath, 'extracted.txt'), 'a') as fout_ext:
		print(f'# extracted using {func_extract.__name__}:', file = fout_ext)
		print(repr(result), file = fout_ext)
	
	return result


def wrap_multi_extract(
		func_extract : ExtractorFunc,
		calc_mean : Callable[
			[Dict[Path,float]], 
			float
		] = lambda x : min(x.values()),
	) -> MultiExtractorFunc:

	def _func_extract_MULTI(
			datadir : Path,
			params : ParamsDict,
			ret_nan : bool = False,
		) -> float:

		# TODO: ret_nan is not actually doing the correct thing here, although its probably unimportant. current implementation does not allow for just one of several processes failing

		extracted : Dict[Path,float] = dict()

		for p in os.listdir(datadir):
			p_joined : Path = joinPath(datadir,p)
			if os.path.isdir(p_joined):
				extracted[p] = func_extract(
					datadir = p_joined,
					params = params,
					ret_nan = ret_nan,
				)

		return calc_mean(extracted)

	# add metadata
	_func_extract_MULTI.__name__ = func_extract.__name__
	_func_extract_MULTI.__doc__ = f"""
		{wrap_multi_extract.__doc__} 
		#### docstring of wrapped function:
		```markdown
		{func_extract.__doc__}
		```
	"""
	return _func_extract_MULTI


"""

 ##### ###### #    # #####  #        ##   ##### ######
   #   #      ##  ## #    # #       #  #    #   #
   #   #####  # ## # #    # #      #    #   #   #####
   #   #      #    # #####  #      ######   #   #
   #   #      #    # #      #      #    #   #   #
   #   ###### #    # #      ###### #    #   #   ######

"""

def extract_TEMPLATE(
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


"""
######## ##     ## ##    ##  ######   ######
##       ##     ## ###   ## ##    ## ##    ##
##       ##     ## ####  ## ##       ##
######   ##     ## ## ## ## ##        ######
##       ##     ## ##  #### ##             ##
##       ##     ## ##   ### ##    ## ##    ##
##        #######  ##    ##  ######   ######
"""


def extract_finalpos(
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
		bodydata : NDArray[Any, CoordsRotArr] = read_body_data(joinPath(datadir,'body.dat'))[-1,0]
		return ( bodydata['x'], bodydata['y'] )

def extract_food_dist(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> float:
	"""extract euclidean distance from head to food
	
	### Returns:
	 - `float` 
	   dist from final head position to food
	"""
	if ret_nan:
		return float('nan')

	# get head pos
	bodydata : NDArray[Any, CoordsRotArr] = read_body_data(joinPath(datadir,'body.dat'))[-1,0]
	pos_head : VecXY = VecXY( bodydata['x'], bodydata['y'] )

	# get food pos
	pos_food : VecXY = VecXY(
		params['ChemoReceptors']['foodPos']['x'],
		params['ChemoReceptors']['foodPos']['y'],
	)

	# return distance
	return dist(pos_head, pos_food)

def extract_food_dist_inv(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> float:
	"""extract inverse of euclidean distance from head to food

	this means that higher value ==> higher fitness
	
	### Returns:
	 - `float` 
	   1/(dist from final head position to food)
	"""
	return 1 / extract_food_dist(datadir, params, ret_nan)


def extract_df_row(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> dict:
	# TODO: implement extracting more data, for parameter sweeps
	raise NotImplementedError('please implement me :(')


def calcmean_symmetric(data : Dict[str,float]) -> float:
	"""gives mean for min out of each pair of angles
	
	VERY FRAGILE!!!"""
	
	# REVIEW: very fragile
	# TODO: make less fragile
	# UGLY: very fragile


	# get the angles and match with pairs
	
	per_angle_lsts : DefaultDict[str, List[float]] = defaultdict(list)
	
	for k,v in data.items():
		# extract data from filename
		# TODO: use params json instead?
		k_dict : Dict[str,str] = dict_from_dirname(k, func_cast = str)

		# take absolute value
		angle : str = k_dict['angle'].strip('- ')

		per_angle_lsts[angle].append(v)

	# TODO: assert 2 elements in each list
	# TODO: assert all angles present
	# TODO: handle angle keys better?

	# min of each pair, then average
	per_angle_min : Dict[str,float] = {
		k : min(lst_v)
		for k,lst_v in per_angle_lsts.items()
	}

	return sum(per_angle_min.values()) / len(per_angle_min.values())


def extract_food_angle_align(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	):
	"""extract mean alignment towards food position
	
	### Parameters:
	 - `datadir : Path`   
	 - `params : ParamsDict`   
	 - `ret_nan : bool`   
	   (defaults to `False`)
	
	### Returns:
	 - `float` 
	"""

	# get head pos
	arr_pos_head : NDArray[Any, CoordsRotArr] = read_body_data(joinPath(datadir,'body.dat'))[:,0]

	# get food pos
	pos_food : VecXY = VecXY(
		params['ChemoReceptors']['foodPos']['x'],
		params['ChemoReceptors']['foodPos']['y'],
	)

	# at each timestep, get angle to food position from current position
	align_angle : NDArray[Any, float] = np.full_like(arr_pos_head['phi'], np.nan)

	if ret_nan:
		return align_angle

	for i,pos_head in enumerate(arr_pos_head):
		# get angle to food
		food_angle : float = angle_between_starr(pos_head, pos_food)
		align_angle[i] = food_angle - pos_head['phi']

	return align_angle


def extract_food_angle_align_mean(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> float:
	"""extract mean alignment towards food position
	
	### Parameters:
	 - `datadir : Path`   
	 - `params : ParamsDict` 
	 - `ret_nan : bool`   
	   (defaults to `False`)
	
	### Returns:
	 - `float` 
	"""
	if ret_nan:
		return float('nan')

	# get the angle
	angle_align : NDArray[Any,float] = extract_food_angle_align(datadir, params)

	# return mean
	return np.mean(np.abs(angle_align))



def extract_gradient_deriv(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	):
	"""compute the change in food concentration at every timestep for head position
	
	### Parameters:
	 - `datadir : Path`   
	   [description]
	 - `params : ParamsDict`   
	   [description]
	 - `ret_nan : bool`   
	   [description]
	   (defaults to `False`)
	
	### Returns:
	 - `NDArray[Any,float]` 
	   [description]
	"""
	# TODO


def extract_combined_grad_angle(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	):
	# TODO
	raise NotImplementedError(f'{extract_combined_grad_angle=}')

