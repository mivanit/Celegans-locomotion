from typing import *
import copy
from math import dist,isnan
import json
import sys
from enum import Enum

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore
from pydbg import dbg # type: ignore

if TYPE_CHECKING:
	from mypy_extensions import Arg
else:
	Arg = lambda t,s : t

__EXPECTED_PATH__ : str = 'pyutil.params'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import *

"""

 ##### #   # #####  # #    #  ####
   #    # #  #    # # ##   # #    #
   #     #   #    # # # #  # #
   #     #   #####  # #  # # #  ###
   #     #   #      # #   ## #    #
   #     #   #      # #    #  ####

"""

Valid_NS = Literal['Head', 'VentralCord']
Valid_Neurons = str
# Valid_Neurons = Literal['Head', 'VentralCord']

class ModTypes(Enum):
	params : str = 'params'
	conn : str = 'conn'
	none : None = None

	# cli : str = 'cli'

# T_ModTypes = Literal[tuple(e.value for e in ModTypes)]
T_ModTypes = Literal['params', 'conn', 'none']

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

NormalDistTuple = NamedTuple(
	'NormalDistTuple', 
	[
		('mu', float), 
		('sigma', float),
	],
)

DistTuple = Union[RangeTuple,NormalDistTuple]

ParamsDict = Dict[str, Any]
ModParamsDict = Dict[ModParam, float]
ModParamsHashable = Tuple[Tuple[ModParam, float], ...]
ModParamsRanges = Dict[ModParam, RangeTuple]
ModParamsDists = Dict[ModParam, DistTuple]


"""

 #    # #  ####   ####
 ##  ## # #      #    #
 # ## # #  ####  #
 #    # #      # #
 #    # # #    # #    #
 #    # #  ####   ####

"""

def distributions_to_ranges(in_data : ModParamsDists, n_sigma : float = 1.5) -> ModParamsRanges:
	output : ModParamsRanges = dict()
	for k,v in in_data.items():
		if isinstance(v, RangeTuple):
			output[k] = v
		elif isinstance(v, NormalDistTuple):
			output[k] = RangeTuple(
				min = v.mu - v.sigma * n_sigma,
				max = v.mu + v.sigma * n_sigma,
			)
		else:
			raise NotImplementedError(f"unknown distribution type:\t{k}\t{v}\t{type(v)}")
	
	return output



def load_params(path : Path) -> ParamsDict:
	with open(path, 'r') as fin:
		data : ParamsDict = json.load(fin)
	return data




def modprmdict_to_filename(
		data : ModParamsDict,
		key_order : Optional[List[ModParam]] = None,
		short_keys : Optional[int] = None,
		delim_pair : str = '_',
		delim_items : str = ',',
	):
	
	if key_order is None:
		key_order = list(data.keys())

	output : List[str] = []

	for k in key_order:
		k_write : str = str(k)
		
		# UGLY: this whole bit
		if k.mod_type == ModTypes.conn.value:
			_,str_from,str_to,_ = k.path.split(',')
			k_write = f"{str_from}-{str_to}".replace('*','x')
		elif k.mod_type == ModTypes.params.value:
			k_write = k.path.split('.')[-1][:short_keys]
		
		output.append(f'{k_write}{delim_pair}{data[k]:.3}')
	
	return delim_items.join(output)

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


	conn_idxs = list()		
	conn_key_temp : dict = deepcopy(conn_key)
	cidx_temp : Optional[int] = None

	# UGLY: clean this bit up

	if conn_key['to'].endswith('*'):
		# if wildcard given, find every connection that matches
		for nrn in params_data[conn_key['NS']]['neurons']:
			# loop over neuron names, check if they match
			# REVIEW: this isnt full regex, but whatever
			if nrn.startswith(conn_key['to'].split('*')[0]):
				conn_key_temp['to'] = nrn
				cidx_temp = find_conn_idx(
					params_data[conn_key_temp['NS']]['connections'],
					conn_key_temp,
				)
				# append to list, but only if an existing connection is found
				# note that this behavior differs from when no wildcard is given,
				# in that new connections will not be created
				if cidx_temp is not None:
					conn_idxs.append(cidx_temp)
					
	elif conn_key['from'].endswith('*'):
		# if wildcard given, find every connection that matches
		for nrn in params_data[conn_key['NS']]['neurons']:
			# loop over neuron names, check if they match
			# REVIEW: this isnt full regex, but whatever
			if nrn.startswith(conn_key['from'].split('*')[0]):
				conn_key_temp['from'] = nrn
				cidx_temp = find_conn_idx(
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

 #    # ###### #####   ####  ######
 ##  ## #      #    # #    # #
 # ## # #####  #    # #      #####
 #    # #      #####  #  ### #
 #    # #      #   #  #    # #
 #    # ###### #    #  ####  ######

"""

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
		if tup_key.mod_type == ModTypes.params.value:
			
			nested_keys : str = tup_key.path

			fin_dic,fin_key = keylist_access_nested_dict(
				d = output, 
				keys = nested_keys.split('.'),
			)
			fin_dic[fin_key] = val

		elif tup_key.mod_type == ModTypes.conn.value:
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

def extract_mods_from_params(
		params : ParamsDict, 
		modkeys : List[ModParam],
		modkeys_striponly : List[ModParam],
		default_val : Any = None,
	) -> Tuple[ParamsDict,ModParamsDict]:
	"""extracts modparams from a params dict (inverts `merge_params_with_mods`)

	returns `Tuple[ParamsDict,ModParamsDict]` containing:
	- copy of the params dict, with extracted values overwritten with `default_val`
	- the mod params dict

	`modkeys` takes precedence over `modkeys_striponly`. is this bad deisgn?

	combining these two using `merge_params_with_mods` should just give back `params`
	"""

	params_stripped : ParamsDict = deepcopy(params)
	modparams : ModParamsDict = dict()

	modkey_set : Set[ModParam] = set(modkeys)

	for key in set([*modkeys, *modkeys_striponly]):
		# merge in the standard params
		if key.mod_type == ModTypes.params.value:
			# access the element
			fin_dic,fin_key = keylist_access_nested_dict(
				d = params_stripped, 
				keys = key.path.split('.'),
			)
			# store copy in modparams
			if key in modkey_set:
				modparams[key] = deepcopy(fin_dic[fin_key])
			# overwrite in params_stripped
			fin_dic[fin_key] = default_val

		elif key.mod_type == ModTypes.conn.value:
			# merge in the connection modifiers

			# get the key
			conn_key = strList_to_dict(
				in_data = key.path,
				keys_list = ['NS', 'from', 'to', 'type'],
				delim = ',',
			)

			# get the indecies of the connections whose weights need to be changed
			conn_idxs : List[Optional[int]] = find_conn_idx_regex(
				params_data = params_stripped,
				conn_key = conn_key,
			)

			for cidx in conn_idxs:
				# store weights in mod params
				if key in modkey_set:
					modparams[key] = (
						params_stripped
						[conn_key['NS']]
						['connections']
						[cidx]
						['weight']
					)

				# strip from params
				params_stripped[conn_key['NS']]['connections'][cidx]['weight'] = default_val

		else:
			raise NotImplementedError(f'given key type {key.mod_type} unknown')

	return params_stripped,modparams



def jointo_nan_eval_runs(eval_runs : List[ModParamsDict]) -> ModParamsDict:
	"""
	create a `ModParamsDict` that maps every key from `eval_runs` to float('nan')
	
	useful for then merging into params
	"""
	joined_nans : ModParamsDict = dict()

	for x in eval_runs:
		for k in x.keys():
			joined_nans[k] = float('nan')

	return joined_nans

	
