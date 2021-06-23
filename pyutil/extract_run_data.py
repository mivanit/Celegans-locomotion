from typing import *
import subprocess
import copy
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

from pyutil.util import *
from pyutil.params import *


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


def _wrap_multi_extract(
		func_extract : ExtractorFunc,
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
	) -> MultiExtractorFunc:

	def _func_extract_MULTI(
			datadir : Path,
			params : ParamsDict,
			ret_nan : bool = False,
		) -> float:

		# TODO: ret_nan is not actually doing the correct thing here, although its probably unimportant. current implementation does not allow for just one of several processes failing

		lst_extracted : List[float] = list()

		for p in os.listdir(datadir):
			p_joined : Path = joinPath(datadir,p)
			if os.path.isdir(p_joined):
				lst_extracted.append(func_extract(
					datadir = p_joined,
					params = params,
					ret_nan = ret_nan,
				))

		return calc_mean(lst_extracted)

	# add metadata
	_func_extract_MULTI.__name__ = func_extract.__name__
	_func_extract_MULTI.__doc__ = f"""
		{_wrap_multi_extract.__doc__} 
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
		bodydata : CoordsRotArr = read_body_data(joinPath(datadir,'body.dat'))[-1,0]
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
	else:
		# get head pos
		bodydata : CoordsRotArr = read_body_data(joinPath(datadir,'body.dat'))[-1,0]
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
