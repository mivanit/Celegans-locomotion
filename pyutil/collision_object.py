import sys
import os
from enum import Enum
from typing import *
from matplotlib.pyplot import install_repl_displayhook

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore

__EXPECTED_PATH__ : str = 'pyutil.collision_object'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import Path


AxBounds = Tuple[float,float]
BoundsKey = Literal[
	'bound_min_x', 'bound_min_y',
	'bound_max_x', 'bound_max_y',
]
BoundingBox = Dict[BoundsKey,float]

def BOUNDS_TEMPLATE() -> BoundingBox:
	return {
		'bound_min_x' : float('inf'),
		'bound_min_y' : float('inf'),
		'bound_max_x' : -float('inf'),
		'bound_max_y' : -float('inf'),
	}


class CollisionType(Enum):
	BASE = None
	Box_Ax = 'Box_Ax'
	Disc = 'Disc'



"""
 ######  ##        ######
##    ## ##       ##    ##
##       ##       ##
##       ##        ######
##       ##             ##
##    ## ##       ##    ##
 ######  ########  ######
"""

# TODO (minor): type hint to `CollisionType` doesn't have expected behavior, this is just because python enum type hints are weird

class CollisionObject(object):
	
	ATTRIBUTES : Dict[CollisionType,List[str]] = {
		CollisionType.BASE : [
			'coll_type', 
			'bound_min_x', 'bound_min_y',
			'bound_max_x', 'bound_max_y',
		],
		CollisionType.Box_Ax : [
			'fvec_x', 'fvec_y',
		],
		CollisionType.Disc : [
			'centerpos_x', 'centerpos_y',
			'force',
			'radius_inner', 'radius_outer',
			'angle_min', 'angle_max'
		]
	}

	TYPECAST : Dict[
		Union[str,None],
		Callable[[str], Any],
	] = {
		None : float,
		'coll_type' : lambda x : CollisionType[x] if isinstance(x,str) else x,
	}

	@staticmethod
	def TYPECAST_FUNC(attr : str) -> Callable[[str], Any]:
		if attr in CollisionObject.TYPECAST:
			return CollisionObject.TYPECAST[attr]
		else:
			return CollisionObject.TYPECAST[None]


	@staticmethod
	def get_attr_list(x : CollisionType):
		if x in CollisionObject.ATTRIBUTES:
			return (
				CollisionObject.ATTRIBUTES[CollisionType.BASE]
				+ CollisionObject.ATTRIBUTES[x]
			)
		else:
			return CollisionObject.ATTRIBUTES[CollisionType.BASE]


	def __init__(self, **kwargs) -> None:
		self.data : Dict[str,Any] = {}

		self.coll_type : CollisionType = kwargs['coll_type']
		if isinstance(kwargs['coll_type'], str):
			self.coll_type = CollisionType[kwargs['coll_type']]

		self.lst_attrs : List[str] = CollisionObject.get_attr_list(self.coll_type)

		# create from list, casting types
		self.data = {
			a : CollisionObject.TYPECAST_FUNC(a)(kwargs[a])
			for a in self.lst_attrs
		}
	
	def __getitem__(self, key : str) -> Any:
		return self.data[key]
	
	def __setitem__(self, key : str, val : Any):
		self.data[key] = val

	# def __getattr__(self, name: str) -> Any:
	# 	if name in self.data:
	# 		return self.data[name]
	# 	else:
	# 		raise AttributeError(name)
	
	# def __setattr__(self, name: str, val : Any):
	# 	if name in self.__dict__:
	# 		self.__dict__[name] = val
	# 	elif name in self.data:
	# 		self.data[name] = val
	# 	else:
	# 		raise AttributeError(name)
			


	def serialize_tsv(self, delim = '\t') -> str:
		output = [self.coll_type.name] + [
			'{:.10}'.format(self.data[a])
			for a in self.lst_attrs[1:]
		]
		return delim.join(output)

	def serialize_lst(self) -> List[Any]:
		output = [self.coll_type.name] + [
			self.data[a]
			for a in self.lst_attrs[1:]
		]
		return output

	def set_force(self, mag : float = 1.0):
		if self.coll_type == CollisionType.Box_Ax:
			umag : float = (self.data['fvec_x']**2.0 + self.data['fvec_y']**2.0)**0.5
			self.data['fvec_x'] *= mag / umag
			self.data['fvec_y'] *= mag / umag

		elif self.coll_type == CollisionType.Disc:
			self.data['force'] = mag * (-1.0 if self.data['force'] < 0 else 1.0)

	def shift_obj(self, shift_x : float = 0.0, shift_y : float = 0.0):
		self.data['bound_min_x'] += shift_x
		self.data['bound_min_y'] += shift_y
		self.data['bound_max_x'] += shift_x
		self.data['bound_max_y'] += shift_y

		if self.coll_type == CollisionType.Disc:
			self.data['centerpos_x'] += shift_x
			self.data['centerpos_y'] += shift_y
	
	def scale_pos(self, scaling_factor : float = 1.0):
		self.data['bound_min_x'] *= scaling_factor
		self.data['bound_min_y'] *= scaling_factor
		self.data['bound_max_x'] *= scaling_factor
		self.data['bound_max_y'] *= scaling_factor

		if self.coll_type == CollisionType.Disc:
			self.data['centerpos_x'] *= scaling_factor
			self.data['centerpos_y'] *= scaling_factor
			self.data['radius_inner'] *= scaling_factor
			self.data['radius_outer'] *= scaling_factor

	
	# def to_poly(self):
	# 	if self.coll_type == CollisionType.Disc:
	# 		n_pts = 50
	# 		theta = np.linspace(
	# 			self.data['angle_min'], self.data['angle_max'], 
	# 			n_pts,
	# 		)
			
	# 		arr_inner = np.concatenate([np.cos(theta), np.sin(theta)], 1)

			
			


	# 	else:
	# 		raise NotImplementedError()


	@staticmethod
	def deserialize_tsv(line : str, delim : str = '\t') -> 'CollisionObject':
		line_lst = line.split(delim)

		# HACK: at some point, the C++ code was producing collision object tsv
		# 	files with an extra tab put in, meaning there was an extra empty 
		# 	string in every line when we split by tabs. hence, this line exists
		# 	to deal with those files. it's safe to remove if you are only 
		# 	dealing with freshly generated files
		while ('' in line_lst):
			line_lst.remove('')

		# read the collision type and the list of attributes
		coll_type : CollisionType = CollisionType[line_lst[0]]
		attrs_lst = CollisionObject.get_attr_list(coll_type)

		# check that we have the expected number of attributes
		assert (len(line_lst) == len(attrs_lst)),f'incorrect number of attributes in line (expected {len(attrs_lst)}, got {len(line_lst)}):\n\t{line}\n\t{line_lst}'

		# read them in
		output : CollisionObject = CollisionObject(**{
			attrs_lst[i] : line_lst[i]
			for i in range(len(attrs_lst))
		})

		return output


	@staticmethod
	def fixkeys_serialized_json(data : Dict[str,Any], delete_old_type : bool = True) -> Dict[str,Any]:
		"""ix a dict read from a hacky json format
		
		why is the key so hacky? because I want to return a simple `<string,double>` hash map in the C++ code, not a `json` object or multi-type hash map. So, we have to iterate thru the keys to find the hash map type.

		We split up the `'__type__'` style keys to get the collision object type, then add that key explicitly to the output and remove the old one

		examples:
		{
			"__type__:Box_Ax": 1.0,
			"bound_max_x": -0.0005,
			"bound_max_y": 0.0054798,
			"bound_min_x": -0.0006,
			"bound_min_y": 0.000979796,
			"coll_type": 0.0,
			"fvec_x": 5e-07,
			"fvec_y": 0.0
		}

		```json
		{
			"coll_type" : "Box_Ax",
			"bound_max_x": -0.0005,
			"bound_max_y": 0.0054798,
			"bound_min_x": -0.0006,
			"bound_min_y": 0.000979796,
			"coll_type": 0.0,
			"fvec_x": 5e-07,
			"fvec_y": 0.0
		}
		```

		```json
		{
			"__type__:Disc": 1.0,
			"angle_max": 1.07915,
			"angle_min": 2.06244,
			"bound_max_x": 0.0012,
			"bound_max_y": 0.0012,
			"bound_min_x": -0.0012,
			"bound_min_y": -0.0012,
			"centerpos_x": 0.0,
			"centerpos_y": 0.0,
			"coll_type": 1.0,
			"force": -5e-07,
			"radius_inner": 0.0011,
			"radius_outer": 0.0012
		}
		```
		"""
		colltype_key : Optional[str] = None
		# first, look for a real key
		# NOTE: this ignores floating point keys that expect to be cast to int
		if ('coll_type' not in data) or (not isinstance(data['coll_type'], str)):
			# get the hacky key
			for key in data:
				if key.startswith('__type__:'):
					if colltype_key is None:
						colltype_key = key.split(':')[1]
					else:
						raise KeyError(f'dict passed to `deserialize_json` has multiple keys starting with `__type__:`, cannot infer `CollisionType`\n\n{data}')
		
			# set the new key
			data['coll_type'] = colltype_key

		if delete_old_type:
			to_del : Set[str] = set()
			for k in data.keys():
				if k.startswith('__type__:'):
					to_del.add(k)
			for k in to_del:
				del data[k]
		
		return data

	@staticmethod
	def deserialize_json_dict(data : Dict[str,Any], delete_old_type : bool = True) -> 'CollisionObject':
		"""deserialize a dict read from a maybe-hacky json format
		
		will call `fixkeys_serialized_json()` if it detects the dict might be hacky

		### Returns:
		 - `CollisionObject` 
		   read object
		"""
		# first, fix the keys if needed
		if 'coll_type' not in data:
			data = CollisionObject.fixkeys_serialized_json(data, delete_old_type)

		return CollisionObject(**data)

	@staticmethod
	def create_box( 
			corner_A,
			corner_B,
			fvec,
		):
		return CollisionObject(
			coll_type = CollisionType.Box_Ax,
			bound_min_x = min(corner_A[0], corner_B[0]), 
			bound_min_y = min(corner_A[1], corner_B[1]),
			bound_max_x = max(corner_A[0], corner_B[0]), 
			bound_max_y = max(corner_A[1], corner_B[1]),
			fvec_x = fvec[0], 
			fvec_y = fvec[1],
		)




"""
########   #######  ##     ## ##    ## ########   ######
##     ## ##     ## ##     ## ###   ## ##     ## ##    ##
##     ## ##     ## ##     ## ####  ## ##     ## ##
########  ##     ## ##     ## ## ## ## ##     ##  ######
##     ## ##     ## ##     ## ##  #### ##     ##       ##
##     ## ##     ## ##     ## ##   ### ##     ## ##    ##
########   #######   #######  ##    ## ########   ######
"""

def get_bounds(collobjs : List[CollisionObject]) -> BoundingBox:
	bounds : BoundingBox = BOUNDS_TEMPLATE()

	for x in collobjs:
		# mins
		for bd in ['bound_min_x', 'bound_min_y']:
			if x[bd] < bounds[bd]:
				bounds[bd] = x[bd]
		# maxes
		for bd in ['bound_max_x', 'bound_max_y']:
			if x[bd] > bounds[bd]:
				bounds[bd] = x[bd]

	return bounds

def _bounds_tuples_to_bbox(bounds_x : AxBounds, bounds_y : AxBounds) -> BoundingBox:
	return {
		'bound_min_x' : bounds_x[0],
		'bound_min_y' : bounds_y[0],
		'bound_max_x' : bounds_x[1],
		'bound_max_y' : bounds_y[1],
	}


def _combine_bounds(lst_bounds : List[BoundingBox]) -> BoundingBox:
	bounds : BoundingBox = BOUNDS_TEMPLATE()

	for x in lst_bounds:
		# mins
		for bd in ['bound_min_x', 'bound_min_y']:
			if x[bd] < bounds[bd]:
				bounds[bd] = x[bd]
		# maxes
		for bd in ['bound_max_x', 'bound_max_y']:
			if x[bd] > bounds[bd]:
				bounds[bd] = x[bd]
	
	return bounds


def get_bbox_ranges(bounds : BoundingBox) -> Tuple[float,float]:
	return (
		bounds['bound_max_x'] - bounds['bound_min_x'],
		bounds['bound_max_y'] - bounds['bound_min_y'],
	)

def pad_BoundingBox(bounds : BoundingBox, pad_frac : float) -> BoundingBox:
	x_range, y_range = get_bbox_ranges(bounds)

	return {
		'bound_min_x' : bounds['bound_min_x'] - x_range * pad_frac,
		'bound_min_y' : bounds['bound_min_y'] - y_range * pad_frac,
		'bound_max_x' : bounds['bound_max_x'] + x_range * pad_frac,
		'bound_max_y' : bounds['bound_max_y'] + y_range * pad_frac,
	}






"""
####  #######
 ##  ##     ##
 ##  ##     ##
 ##  ##     ##
 ##  ##     ##
 ##  ##     ##
####  #######
"""

def save_collobjs_tsv(
		lst_data : List[CollisionObject],
		filename : Path = "../data/collision_objs.tsv",
	):
	with open(filename, 'w') as fout:
		fout.write('\n'.join([
			x.serialize_tsv()
			for x in lst_data
		]))


def read_collobjs_tsv(filename : Path = "../data/collision_objs.tsv") -> List[CollisionObject]:
	output : List[CollisionObject] = []
	with open(filename, 'r') as fin:
		for line in fin:
			output.append(CollisionObject.deserialize_tsv(line))
	
	return output


"""
##       ########  ######      ###     ######  ##    ##
##       ##       ##    ##    ## ##   ##    ##  ##  ##
##       ##       ##         ##   ##  ##         ####
##       ######   ##   #### ##     ## ##          ##
##       ##       ##    ##  ######### ##          ##
##       ##       ##    ##  ##     ## ##    ##    ##
######## ########  ######   ##     ##  ######     ##
""" 


def read_legacy_collobjs(objs_file : str) -> Tuple[NDArray,NDArray]:
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