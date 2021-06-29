"""
be sure to run from the directory this is in, if not importing as a package
"""
from typing import *
import sys

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore

import numba
from pydbg import dbg # type: ignore

if __name__ == '__main__':
	sys.path.append('../..')
from pyutil.util import *


def original(filename : Path) -> NDArray[(Any,Any), CoordsRotArr]:
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


def numpy_optimized(filename : Path) -> NDArray[(Any,Any), CoordsRotArr]:
	# read in
	data_raw : NDArray = np.genfromtxt(filename, delimiter = ' ', dtype = None)

	# trim first variable (time)
	data_raw = data_raw[:,1:]

	# compute dims
	n_tstep : int  = data_raw.shape[0]
	n_seg : int  = data_raw.shape[1] // 3

	# reshape to allow casting to structured array
	data_raw = np.reshape(data_raw, (n_tstep, n_seg, 3))

	return recfunctions.unstructured_to_structured(
		data_raw,
		dtype = CoordsRotArr,
	)

@numba.njit(cache=True)
def _numba_process_A(data_raw : NDArray[Any, float]) -> NDArray[Any, CoordsRotArr]:
	# trim first variable (time)
	data_raw = data_raw[:,1:].copy()

	# compute dims
	n_tstep : int  = data_raw.shape[0]
	n_seg : int  = data_raw.shape[1] // 3

	# reshape to allow casting to structured array
	return np.reshape(data_raw, (n_tstep, n_seg, 3))

def numba_optimized_A(filename : Path) -> NDArray[(Any,Any), CoordsRotArr]:
	"""doesnt work, numba doesnt implement recfuncs :( """
	# read in
	data_raw : NDArray = np.genfromtxt(filename, delimiter = ' ', dtype = None)

	# process
	return recfunctions.unstructured_to_structured(
		_numba_process_A(data_raw),
		dtype = CoordsRotArr,
	)

TUP_CRA = ('x', 'y', 'phi')

@numba.njit(cache=True)
def _numba_process_B(data_raw : NDArray[Any, float]) -> NDArray[Any, CoordsRotArr]:
	# trim first variable (time)
	data_raw = data_raw[:,1:].copy()

	# compute dims
	n_tstep : int = data_raw.shape[0]
	n_seg : int  = data_raw.shape[1] // 3

	# reshape to allow casting to structured array
	data_raw = np.reshape(data_raw, (n_tstep, n_seg, 3))

	return data_raw

	# data : NDArray[(n_tstep, n_seg), CoordsRotArr] = np.empty(
	# 	shape = (n_tstep, n_seg),
	# 	dtype = CoordsRotArr,
	# )
	
	# # organize by x pos, y pos, and rotation (phi)
	# data[:, :][TUP_CRA[0]] = data_raw[:, :, 0]
	# data[:, :][TUP_CRA[1]] = data_raw[:, :, 1]
	# data[:, :][TUP_CRA[2]] = data_raw[:, :, 2]

	# return data

def numba_optimized_B(filename : Path) -> NDArray[(Any,Any), CoordsRotArr]:
	"""doesnt work, numba doesnt implement recfuncs :( """
	# read in
	data_raw : NDArray = np.genfromtxt(filename, delimiter = ' ', dtype = None)

	# process
	return _numba_process_B(data_raw)



READFUNCS_PROFILE = [
	original,
	numpy_optimized,
	numba_optimized_A,
	numba_optimized_B,
]



def run_prof(datfile : Path) -> Dict[str,float]:
	import timeit

	data_compare : Dict[str,NDArray] = dict()
	output : Dict[str,float] = dict()

	for func in READFUNCS_PROFILE:
		func_name : str = func.__name__
		data_compare[func_name] = func(datfile)
		print(f'> profling {func_name}')
		output[func_name] = timeit.timeit(lambda : func(datfile), number = 100)


	for k,v in data_compare.items():
		print(f'{k:<20}\t{v.shape}\t{v.dtype}')

	try:
		print('comparison:')
		for fl in CoordsRotArr.names:
			print(
				f'\t{fl}:\t',
				[
					np.all(np.equal(
						v[fl],
						data_compare[READFUNCS_PROFILE[0].__name__][fl]
					))
					for k,v in data_compare.items()
				]
			)
	except IndexError:
		print('failed comparison!')
	
	return output

if __name__ == '__main__':
	import fire
	fire.Fire(run_prof)
