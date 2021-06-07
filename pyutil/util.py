import os
import math
from typing import *
from copy import deepcopy
from enum import Enum
# from collections import namedtuple
# from dataclasses import dataclass

import json

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore

"""
 #####    ##   ##### #    #
 #    #  #  #    #   #    #
 #    # #    #   #   ######
 #####  ######   #   #    #
 #      #    #   #   #    #
 #      #    #   #   #    #
"""
Path = str

def mkdir(p : Path):
	if not os.path.isdir(p):
		os.mkdir(p)

def joinPath(*args):
	return os.path.join(*args).replace("\\", "/")

# def joinPath(*args):
# 	output : Path = '/'.join(args).replace("\\", "/")
# 	while '//' in output:
# 		output.replace("//", "/")	
# 	return output

"""
 #    # #  ####   ####
 ##  ## # #      #    #
 # ## # #  ####  #
 #    # #      # #
 #    # # #    # #    #
 #    # #  ####   ####
"""

CoordsArr = np.dtype([ ('x','f8'), ('y','f8')])
CoordsRotArr = np.dtype([ ('x','f8'), ('y','f8'), ('phi','f8') ])

VecXY = NamedTuple(
	'VecXY',
	[
		('x', float), 
		('y', float),
	],
)

def dump_state(dict_locals : dict, path : Path, file : Path = 'locals.txt'):
	with open(joinPath(path, file), 'w') as log_out:
		# json.dump(dict_locals, log_out, indent = '\t')
		print(dict_locals, file = log_out)


def prntmsg(msg : str, indent = 0):
	print(f"{'  '*indent}> {msg}")

"""
########  ####  ######  ########
##     ##  ##  ##    ##    ##
##     ##  ##  ##          ##
##     ##  ##  ##          ##
##     ##  ##  ##          ##
##     ##  ##  ##    ##    ##
########  ####  ######     ##
"""

def keylist_access_nested_dict(
		d : Dict[str,Any], 
		keys : List[str],
	) -> Tuple[dict,str]:
	
	fin_dict : dict = d
	for k in keys[:-1]:
		fin_dict = fin_dict[k]
	fin_key = keys[-1]

	return (fin_dict,fin_key)


def split_dict_arrs(in_dict : Dict[float,float]):
	return zip(*sorted(in_dict.items()))


def strList_to_dict(
		in_data : Union[dict,tuple,str], 
		keys_list : List[str], 
		delim : str = ',',
		type_map : Dict[str,Callable] = dict(),
	) -> Dict[str,Any]:
	if isinstance(in_data, dict):
		return in_data
	else:
		in_lst : Optional[List[str]] = None
		if isinstance(in_data, tuple):
			in_lst = list(in_data)
		elif isinstance(in_data, str):
			# split into list
			in_lst = in_data.split(delim)
		else:
			raise TypeError(f'invalid type, expected one of dict,tuple,str, got: {type(in_data)} for {in_data}')

		# map to the keys
		out_dict : Dict[str,Any] = {
			k : v 
			for k,v in zip(keys_list, in_lst)
		}

		# map types
		for key,func in type_map.items():
			if key in out_dict:
				out_dict[key] = func(out_dict[key])
		
		return out_dict

def dict_to_filename(
		data : Dict[str,float], 
		key_order : Optional[List[str]] = None,
		short_keys : Optional[int] = None,
	) -> str:

	if key_order is None:
		key_order = list(data.keys())

	output : List[str] = []

	for k in key_order:
		# shorten the keys by splitting by dot, 
		# and taking the first `short_keys` chars of the last bit
		k_short : str = k.split('.')[-1][:short_keys]
		output.append(f'{k_short}={data[k_short]:.3}')
	
	return '_'.join(output)

def dict_hash(data : dict, hash_len_mod : int = int(10**8)) -> int:
	return hash(tuple(data.items())) % hash_len_mod


"""

 #    #  ####  #####     #####    ##   #####    ##   #    #  ####
 ##  ## #    # #    #    #    #  #  #  #    #  #  #  ##  ## #
 # ## # #    # #    #    #    # #    # #    # #    # # ## #  ####
 #    # #    # #    #    #####  ###### #####  ###### #    #      #
 #    # #    # #    #    #      #    # #   #  #    # #    # #    #
 #    #  ####  #####     #      #    # #    # #    # #    #  ####

"""

Valid_NS = Literal['Head', 'VentralCord']
Valid_Neurons = str
# Valid_Neurons = Literal['Head', 'VentralCord']

class ModTypes(Enum):
	params : str = 'params'
	conn : str = 'conn'
	# cli : str = 'cli'

# T_ModTypes = Literal[tuple(e.value for e in ModTypes)]
T_ModTypes = Literal['params', 'conn']

# ModTypes = Literal[
# 	'params',
# 	'conn',
# 	'cli',
# ]

ModParam = NamedTuple(
	'ModParam', 
	[
		('mod_type', T_ModTypes), 
		('path', str),
	],
)

RangeTuple = NamedTuple(
	'RangeTuple', 
	[
		('min', float), 
		('max', float),
	],
)

ParamsDict = Dict[str, Any]
ModParamsDict = Dict[ModParam, float]
ModParamsRanges = Dict[ModParam, RangeTuple]

MODPARAMS_DEFAULT_RANGES : ModParamsRanges = {
	ModParam("conn",   "Head,AWA,RIM,chem") : RangeTuple(-40000,40000),
	ModParam("conn",   "Head,RIM,RMD*,chem") : RangeTuple(-100,100),
	# ModParam("params", "ChemoReceptors.kappa") : RangeTuple(150.0, 250.0),
	# ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-300000, 400000),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

def load_params(path : Path) -> ParamsDict:
	with open(path, 'r') as fin:
		data : ParamsDict = json.load(fin)
	return data

# ConnKey = NamedTuple(
# 	'ConnKey',
# 	[
# 		('NS', Valid_NS),
# 		('from', Valid_Neurons),
# 		('to', Valid_Neurons),
# 		('weight', float),
# 	],
# )


"""
 ######   #######  ##    ## ##    ##
##    ## ##     ## ###   ## ###   ##
##       ##     ## ####  ## ####  ##
##       ##     ## ## ## ## ## ## ##
##       ##     ## ##  #### ##  ####
##    ## ##     ## ##   ### ##   ###
 ######   #######  ##    ## ##    ##
"""

def find_conn_idx(params_data : Dict[str,Any], conn_key : dict) -> Optional[int]:
	"""finds the index of the entry matching conn_key"""

	for i,item in enumerate(params_data):
		if all([
				conn_key[k] == item[k]
				for k in conn_key 
				if k != 'NS'
			]):
			return i

	return None


def find_conn_idx_regex(
		params_data : Dict[str,Any], 
		conn_key : dict,
		# special_scaling_map : Optional[Dict[str,float]] = None,
	) -> List[Optional[int]]:

	conn_idxs : List[Optional[int]] = [None]

	if conn_key['to'].endswith('*'):
		# if wildcard given, find every connection that matches
		conn_idxs = list()
		
		conn_key_temp : dict = deepcopy(conn_key)
		
		for nrn in params_data[conn_key['NS']]['neurons']:
			# loop over neuron names, check if they match
			# REVIEW: this isnt full regex, but whatever
			if nrn.startswith(conn_key['to'].split('*')[0]):
				conn_key_temp['to'] = nrn
				cidx_temp : Optional[int] = find_conn_idx(
					params_data[conn_key_temp['NS']]['connections'],
					conn_key_temp,
				)
				# append to list, but only if an existing connection is found
				# note that this behavior differs from when no wildcard is given,
				# in that new connections will not be created
				if cidx_temp is not None:
					conn_idxs.append(cidx_temp)
	else:
		# if special_scaling_map is not None:
		# 	raise ValueError(f"`special_scaling_map` specified, but no wildcard given in neuron name:   {special_scaling_map}    {conn_key['to']}")

		# if no wildcard specified, just get the one connection
		conn_idxs = [ find_conn_idx(
			params_data[conn_key['NS']]['connections'],
			conn_key,
		) ]
	
	return conn_idxs



"""
 ######  ##     ## ########
##    ## ###   ### ##     ##
##       #### #### ##     ##
##       ## ### ## ##     ##
##       ##     ## ##     ##
##    ## ##     ## ##     ##
 ######  ##     ## ########
"""

SCRIPTNAME_KEY = "__main__"
COMMAND_DANGERS = [';', 'rm', 'sudo']

def _make_cmd_arg(arg : str, val : Optional[Any]) -> str:
	if val is None:
		return ""
	else:
		return f"--{arg} {val}"
	


def _command_assembler(**kwargs) -> str:
	output : List[str] = [ kwargs[SCRIPTNAME_KEY] ]

	for key,val in kwargs.items():
		if key != SCRIPTNAME_KEY:
			if val is not None:
				output.append(_make_cmd_arg(key, val))

	str_output : str = " ".join(output)
	
	for d in COMMAND_DANGERS:
		if d in str_output:
			print(f'WARNING: command contains: "{d}"\nplease review generated command:\n\t{str_output}\n')
			bln_cont : str = input('continue? (y/n)')
			if bln_cont != 'y':
				exit(0)

	return str_output


def genCmd_singlerun(
		params : Optional[Path] = None,
		coll : Optional[Path] = None,
		output : Optional[Path] = None,
		angle : Optional[float] = None,
		duration : Optional[float] = None,
		foodPos : Union[str, Tuple[float,float], None] = None,
		rand : Optional[bool] = None,
		seed : Optional[int] = None,
	) -> str:
	"""gets a shell command string for launching singlerun
	
	`./singlerun.exe <FLAGS>`
	
	### Parameters:
	 - `params : Optional[Path]`   
	   params json file
	   (defaults to `None`)
	 - `coll : Optional[Path]`   
	   collision tsv file
	   (defaults to `None`)
	 - `output : Optional[Path]`   
	   output dir
	   (defaults to `None`)
	 - `angle : Optional[float]`   
	   starting angle
	   (defaults to `None`)
	 - `duration : Optional[float]`   
	   sim duration in seconds
	   (defaults to `None`)
	 - `foodPos : Union[str, Tuple[float,float], None]`   
	   food position (comma separated) (defaults to whatever is in params.json). set to "DISABLE" to set the scalar to zero
	   (defaults to `None`)
	 - `rand : Optional[bool]`   
	   random initialization seed based on time
	   (defaults to `None`)
	 - `seed : Optional[int]`   
	   set random initialization seed. takes priority over `rand`. seed is 0 by default
	   (defaults to `None`)
	
	### Returns:
	 - `str` 
	   shell command
	"""

	cmd : str = _command_assembler(**{
		SCRIPTNAME_KEY : "./singlerun.exe",
		**locals(),
	}) 

	return cmd
	# return cmd + f' > {output}log.txt'





"""
########  ########    ###    ########
##     ## ##         ## ##   ##     ##
##     ## ##        ##   ##  ##     ##
########  ######   ##     ## ##     ##
##   ##   ##       ######### ##     ##
##    ##  ##       ##     ## ##     ##
##     ## ######## ##     ## ########
"""


def read_body_data(filename : Path) -> NDArray[(Any,Any), CoordsRotArr]:
	"""reads given tsv file into a numpy array
	
	array is a 2-D structured array of type `CoordsRotArr`
	with `'x', 'y', 'phi'` fields for each segment
	so essentially 3-D, where first index is timestep, second is segment, and third/field is x/y/phi
	
	### Parameters:
	- `filename : Path`   
	filename to read
	
	### Returns:
	- `NDArray[Any, CoordsRotArr]` 
	"""
	# read in
	data_raw : NDArray = np.genfromtxt(filename, delimiter = ' ', dtype = None)

	# trim first variable (time)
	data_raw = data_raw[:,1:]

	# compute dims
	n_tstep = data_raw.shape[0]
	n_seg = int(data_raw.shape[1] / 3)

	# allocate new array
	# data : NDArray[(n_tstep, n_seg), CoordsRotArr] = np.full(
	data : NDArray[(n_tstep, n_seg)] = np.full(
		shape = (n_tstep, n_seg),
		fill_value = np.nan,
		dtype = CoordsRotArr,
	)

	# organize by x pos, y pos, and rotation (phi)
	for s in range(n_seg):
		data[:, s]['x'] = data_raw[:, s*3]
		data[:, s]['y'] = data_raw[:, s*3 + 1]
		data[:, s]['phi'] = data_raw[:, s*3 + 2]

	return data


def read_coll_objs_file(objs_file : str) -> Tuple[NDArray,NDArray]:
	"""reads an old blocks/vecs style collider file
	
	### Parameters:
	 - `objs_file : str`   
	
	### Returns:
	 - `Tuple[NDArray,NDArray]` 
	"""
	blocks : list = []
	vecs : list = []
	
	with open(objs_file, 'r') as fin:
		for row in fin:
			row_lst : List[float] = [
				float(x) 
				for x in row.strip().split()
			]

			blocks.append([ row_lst[0:2], row_lst[2:4] ])
			vecs.append(row_lst[4:])

	return (np.array(blocks), np.array(vecs))
