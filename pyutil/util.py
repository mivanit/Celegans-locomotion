from __future__ import annotations
from collections import namedtuple

import os
import sys
import math
from typing import *
from copy import Error, deepcopy
from enum import Enum
import glob
import inspect
# from collections import namedtuple
# from dataclasses import dataclass
from functools import wraps as functools_wraps
import json

# from pydbg import dbg # type: ignore

import numpy as np # type: ignore
import numpy.lib.recfunctions as recfunctions
from nptyping import NDArray # type: ignore

import pandas as pd # type: ignore

# import numba

"""
 #####    ##   ##### #    #
 #    #  #  #    #   #    #
 #    # #    #   #   ######
 #####  ######   #   #    #
 #      #    #   #   #    #
 #      #    #   #   #    #
"""
# Path = str

class Path(str):
	def __truediv__(self, other : Path):
		if other.startswith('/'):
			raise ValueError(f'trying to append rootpath {other} to path')
		if self.endswith('/'):
			return Path(self + other)
		else:
			return Path(self + '/' + other)

	def unixPath(self):
		return Path(self.replace('\\', '/'))

def unixPath(in_path : Path) -> Path:
	return Path(in_path.replace("\\", "/"))

def mkdir(p : Path):
	if not os.path.isdir(p):
		os.mkdir(p)

def joinPath(*args) -> Path:
	return os.path.join(*args).replace("\\", "/")

def get_last_dir_name(p : Path, i_from_last : int = -1) -> Path:
	return unixPath(p).strip('/').split('/')[i_from_last]

def read_file(path : Path) -> str:
	with open(path, 'r') as f:
		return f.read()

# def joinPath(*args):
# 	output : Path = '/'.join(args).replace("\\", "/")
# 	while '//' in output:
# 		output.replace("//", "/")	
# 	return output

# GeneRunID = NamedTuple(
# 	'GeneRunID',
# 	[
# 		('gen', int), 
# 		('h', int),
# 	],
# )


def get_dirs_containing_file(rootdir : Path, wanted_file : Path) -> List[Path]:
	"""returns a list of all dirs in `rootdir` containing `wanted_file`
	
	### Parameters:
	 - `rootdir : Path`   
	 - `wanted_file : Path`   
	
	### Returns:
	 - `List[Path]` 
	
	### Raises:
	 - `FileNotFoundError` : if no suitable directories found
	"""	

	if not isinstance(rootdir, Path):
		rootdir = Path(rootdir)
	if not isinstance(wanted_file, Path):
		wanted_file = Path(wanted_file)

	# check for wildcards in `rootdir`
	rootdir = unixPath(rootdir)
	if '*' in rootdir:
		rootdir_split : List[Path] = rootdir.split('/')
		
		# check wildcards only in the final part of the path
		if (
				all(c == '*' for c in rootdir_split[-1])
				and all('*' not in x for x in rootdir_split[:-1])
			):
			# get rid of them, cause we will add them later
			rootdir = '/'.join(rootdir_split[:-1])
		else:
			Warning(f'unexpected wildcard in rootdir, this might break: {rootdir=}')

	# get all instances of the wanted file
	lst_wanted : List[Path] = glob.glob(rootdir / Path('**') / wanted_file, recursive = True)
	# get the containing directories
	lst_dirs : List[Path] = [ 
		unixPath(os.path.dirname(p)).rstrip('/') + '/'
		for p in lst_wanted
	]

	# make the error less confusing
	if len(lst_dirs) == 0:
		raise FileNotFoundError(f'Could not find any matching files: \n\t{rootdir=}\n\t{lst_wanted=}\n\t{lst_dirs=}')

	return lst_dirs

def deco_str_to_path_kwargs(keywords : Iterable[str], do_posargs : bool = False) -> Callable[[Callable], Callable]:
	"""wraps `func` such that args in `keywords` are converted to `Path` (from `str`)
	
	### Parameters:
	 - `keywords : Iterable[str]`
	 	keywords to convert
	 - `do_posargs : bool`
	 	whether to also convert (all) positional arguments
	   (default: `False`)
	
	### Returns:
	 - `Callable[[Callable], Callable]` 
	"""	

	def _deco(func : Callable) -> Callable:

		@functools_wraps(func)
		def wrapped(*args, **kwargs):
			
			print(f'converting: {keywords=} {do_posargs=} {args=} {kwargs=}')

			for kw in keywords:
				if kw in kwargs:
					kwargs[kw] = Path(kwargs[kw])
			
			if do_posargs:
				args = tuple( Path(x) for x in args )

			print(f'converted: {tuple(type(x) for x in args)=} {args=} {kwargs=}')

			return func(*args, **kwargs)
		
		return wrapped
	
	return _deco

class GeneRunID(NamedTuple):
	gen : int
	h : int

	def __repr__(self) -> str:
		return f"g{self.gen}/h{self.h}"
	
	@staticmethod
	def from_str(p : str) -> GeneRunID:
		p_split : List[str] = unixPath(p).strip('/').split('/')[:2]

		return GeneRunID(
			gen = int(p_split[0].strip('gen_/ ')),
			h = int(p_split[1].strip('h/ ')),
		)
		



"""
 #    # #  ####   ####
 ##  ## # #      #    #
 # ## # #  ####  #
 #    # #      # #
 #    # # #    # #    #
 #    # #  ####   ####
"""

ShapeAnnotation = NewType("ShapeAnnotation", Tuple[str,...])

CoordsArr = np.dtype([ ('x','f8'), ('y','f8')])
CoordsRotArr = np.dtype([ ('x','f8'), ('y','f8'), ('phi','f8') ])

# CoordsArrUnion = Union[CoordsArr, CoordsRotArr]

VecXY = NamedTuple(
	'VecXY',
	[
		('x', float), 
		('y', float),
	],
)

def angle_between_VecXY(u : VecXY, v : VecXY) -> float:
	"""compute the angle from point `u` to point `v`"""
	return np.arctan2(v.y - u.y, v.x - u.x)

def angle_between_starr(u : CoordsArr, v : CoordsArr) -> float:
	"""compute the angle from point `u` to point `v`"""
	return np.arctan2(v['y'] - u['y'], v['x'] - u['x'])

def dump_state(dict_locals : dict, path : Path, file : Path = 'locals.txt'):
	with open(joinPath(path, file), 'w') as log_out:
		# json.dump(dict_locals, log_out, indent = '\t')
		print(dict_locals, file = log_out)


def prntmsg(msg : str, indent = 0):
	print(f"{'  '*indent}> {msg}")


def norm_prob(arr : NDArray) -> NDArray:
	sum_arr : float = np.sum(arr)
	if sum_arr > 0:
		return arr / sum_arr
	else:
		return np.ones(arr.shape, dtype = arr.dtype) / len(arr)


def raise_(ex):
    raise ex

_ExpType = TypeVar('_ExpType')
def pdbg(exp: _ExpType) -> _ExpType:
    for frame in inspect.stack():
        line = frame.code_context[0]
        if "pdbg" in line:
            start = line.find('(') + 1
            end =  line.rfind(')')
            if end == -1:
                end = len(line)
            print(
                f"[{os.path.basename(frame.filename)}:{frame.lineno}] {line[start:end]} = {exp!r}",
                file=sys.stderr,
            )
            break

    return exp


def isinstance_namedtuple(x):
	"""checks if `x` is a `namedtuple`

	credit to https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
	"""
	t = type(x)
	b = t.__bases__
	if len(b) != 1 or b[0] != tuple:
		return False
	f = getattr(t, '_fields', None)
	if not isinstance(f, tuple):
		return False
	return all(type(n)==str for n in f)


"""
########  ####  ######  ########
##     ##  ##  ##    ##    ##
##     ##  ##  ##          ##
##     ##  ##  ##          ##
##     ##  ##  ##          ##
##     ##  ##  ##    ##    ##
########  ####  ######     ##
"""

def wrapper_printdict(func : Callable[..., dict]):
	
	def newfunc(*args, **kwargs) -> None:
		data : dict = func(*args, **kwargs)
		for x in data:
			print(f'{x}\t{data[x]}')

	# add metadata
	newfunc.__name__ = func.__name__
	newfunc.__doc__ = f"""
		{wrapper_printdict.__doc__} 
		#### docstring of wrapped function:
		```markdown
		{func.__doc__}
		```
	"""
	
	return newfunc

def keylist_access_nested_dict(
		d : Dict[str,Any], 
		keys : List[str],
	) -> Tuple[dict,str]:
	"""given a keylist `keys`, return (x,y) where x[y] is d[keys]

	by pretending that `d` can be accessed dotlist-style, with keys in the list being keys to successive nested dicts, we can provide both read and write access to the element of `d` pointed to by `keys`
	
	### Parameters:
	 - `d : Dict[str,Any]`   
	   dict to access
	 - `keys : List[str]`   
	   list of keys to nested dict `d`
	
	### Returns:
	 - `Tuple[dict,str]` 
	   dict is the final layer dict which contains the element pointed to by `keys`, and the string is the last key in `keys`
	"""
	
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
		delim_pair : str = '_',
		delim_items : str = ',',
	) -> str:
	"""convert a dictionary to a filename
	
	format:
	`key_value,otherkey_otherval,morekey_-0.1`
	dont do this with long dicts, and dont use unsafe keys!
	if `short_keys` is true, trims each key to that many chars
	
	### Parameters:
	 - `data : Dict[str,float]`   
	   dict to turn into a filename. if keys are dotlists, use the last element of the dotlist
	 - `key_order : Optional[List[str]]`   
	   if specfied, list the keys in this order
	   (defaults to `None`)
	 - `short_keys : Optional[int]`   
	   if specified, shorten each key to this many chars
	   (defaults to `None`)
	
	### Returns:
	 - `str` 
	   string from the dict
	"""

	if key_order is None:
		key_order = list(data.keys())

	output : List[str] = []

	for k in key_order:
		# shorten the keys by splitting by dot, 
		# and taking the first `short_keys` chars of the last bit
		k_short : str = k.split('.')[-1][:short_keys]
		output.append(f'{k_short}{delim_pair}{data[k_short]:.3}')
	
	return delim_items.join(output)

def dict_from_dirname(
		name : str, 
		func_cast : Callable[[str], Any] = float,
		delim_pair : str = '_',
		delim_items : str = ',',
	) -> Dict[str,Any]:
	"""	this is an ugly ugly hack, dont use it
	
	doesnt handle file extensions (among other things)
	"""
	lst_items : List[str] = name.strip(' /\\').split(delim_items)
	
	output : Dict[str,Any] = dict()

	for x in lst_items:
		x_spl : List[str] = x.split(delim_pair)
		output[x_spl[0]] = func_cast(x_spl[1])

	return output
		


def dict_hash(data : dict, hash_len_mod : int = int(10**8)) -> int:
	""""hashes" a dict in a non-recoverable way
	
	used mainly for uniquely naming files/dirs
	
	### Parameters:
	 - `data : dict`   
	 - `hash_len_mod : int`   
	   (defaults to `int(10**8)`)
	
	### Returns:
	 - `int`
	"""	
	
	return hash(tuple(data.items())) % hash_len_mod


def arbit_obj_serializer_hashable_4json(obj : Any) -> Union[bool,int,float,str]:
	"""serialize an object to a hashable type for json serialization
	
	### Parameters:
	 - `obj : Any`   
	
	### Returns:
	 - `Union[bool,int,float,str]`
	"""	
	
	if isinstance(obj, (bool,int,float,str)):
		return obj
	else:
		return str(obj)




SERIALIZER_SPECIAL_KEYS : List[str] = [
	'__name__',
	'__doc__',
	'__module__',
	'__class__',
]

SERIALIZER_SPECIAL_FUNCS : Dict[str,Callable] = {
	'str' : str,
	'type' : lambda x : type(x).__name__,
	'repr' : lambda x : repr(x),
	'code' : lambda x : inspect.getsource(x),
	'sourcefile' : lambda x : inspect.getsourcefile(x),
}

def arbit_obj_serializer_4json(obj : Any, depth : int = -1 ) -> Any:
	
	try:
		# if primitive type, just add it
		if isinstance(obj, (bool,int,float,str)):
			return obj

		# if max depth is reached, return the object as a string and dont recurse
		if depth == 0:
			return str(obj)
		
		if isinstance(obj, dict):
			# if dict, recurse
			out_dict : Dict[str,Any] = dict()
			for k,v in obj.items():
				out_dict[str(k)] = arbit_obj_serializer_4json(v, depth-1)
			return out_dict

		elif isinstance_namedtuple(obj):
			# if namedtuple, treat as dict
			return arbit_obj_serializer_4json(dict(obj._asdict()))

		elif isinstance(obj, (set,list,tuple)):
			# if iterable, recurse
			return [
				arbit_obj_serializer_4json(x) for x in obj
			]

		else:
			# if not basic type, serialize it
			return {
				**{
					k : str(getattr(obj, k, None))
					for k in SERIALIZER_SPECIAL_KEYS
				},
				**{
					k : str(f(obj))
					for k,f in SERIALIZER_SPECIAL_FUNCS.items()
				},
				# **{
				# 	str(k) : str(v) if str(k).startswith('_')
				# 	else arbit_obj_serializer_4json(v)
				# 	for k,v in obj.__dict__.items()
				# },
			}
	except Exception as e:
		# print(f'error serializing {obj}')
		return str(obj)

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

COMMAND_ARGS_CONVERTERS : Dict[str,Callable] = {
	'foodPos' : lambda x : ','.join(str(y) for y in x) if isinstance(x,tuple) else x,
}


def _make_cmd_arg(arg : str, val : Optional[Any]) -> str:
	if val is None:
		return ""
	else:
		return f"--{arg} {val}"
	


def _command_assembler(**kwargs) -> str:
	output : List[str] = [ kwargs[SCRIPTNAME_KEY] ]

	# convert keyword args, if required
	for k,v in kwargs.items():
		if k in COMMAND_ARGS_CONVERTERS:
			kwargs[k] = COMMAND_ARGS_CONVERTERS[k](v)

	# filter items to remove what's not needed
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
	"""gets a shell command string for launching sim
	
	`./sim.exe <FLAGS>`
	
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

	# REVIEW: is using '**locals()' a good idea here?
	# REVIEW: `duration`, possily other things not properly overriding defaults from loaded params.json

	cmd : str = _command_assembler(**{
		SCRIPTNAME_KEY : "./sim.exe",
		**locals(),
	}) 

	return cmd
	# return cmd + f' > {output}log.txt'

