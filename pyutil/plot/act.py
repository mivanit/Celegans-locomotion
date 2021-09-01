from typing import *
import re
import os

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore

import matplotlib.pyplot as plt # type: ignore

import pandas as pd # type: ignore


__EXPECTED_PATH__ : str = 'pyutil.plot.act'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import Path,joinPath
from pyutil.read_runs import read_act_data

def pattern_match_names(
		names : Optional[List[str]],
		valid_names : List[str],
		match_regex : bool = True,
		remove_dupes : bool = True,
	) -> List[str]:
	
	# if none given, just do every column
	if names is None:
		return valid_names

	# pattern matching
	names_new : List[str] = []
	for x in names:
		if x in valid_names:
			# if its in there, just return that
			names_new.append(x)
		elif match_regex:
			# if not, try to match with regex
			try:
				x_searchstr : Pattern = re.compile(x)
				for y in valid_names:
					if re.match(x_searchstr, y) is not None:
						names_new.append(y)
			except re.error as ex:
				print(ex)
				print(f'  >>  not valid regex, skipping:\t{x}')
		else:
			print(f'  >>  not valid neuron name, skipping:\t{x}')

	if remove_dupes:
		names_new = list( dict.fromkeys(names_new) )
	
	return names_new


def plot_act(
		rootdir : str = 'data/run/act.dat',
		names : Union[str,List[str],None] = None, 
		strict_fname : bool = True,
		show : bool = True,
	):
	"""plot activations of worm neurons
	
	### Parameters:
	 - `rootdir : str`   
	   file to look for, expects tsv forums
	   (defaults to `'data/run/act.dat'`)
	 - `names : Union[str,List[str],None]`   
	   comma-separated (or regular) list of strings. will attempt to match using regex
	   (defaults to `None`)
	 - `strict_fname : bool`   
	   set this to false if you want to use a rootdir other than `'act.dat'`
	   (defaults to `True`)
	"""

	# fix rootdir if only dir given
	if rootdir.endswith('/') or (strict_fname and not rootdir.endswith('act.dat')):
		rootdir = joinPath(rootdir, 'act.dat')
	
	print(rootdir)

	# split names
	if isinstance(names,str):
		names = names.split(',')

	print(f'> raw names: {names}')

	# read data
	data_raw = pd.read_csv(rootdir, sep = ' ').to_records(index=False)
	fields_raw : List[str] = list(data_raw.dtype.fields.keys())
	# data_raw = np.genfromtxt(rootdir, delimiter = ' ', dtype = np.float).T
	# print(data_raw.shape, fields_raw)

	names_new : List[str] = pattern_match_names(names, fields_raw)
	
	# dont plot the time
	if 't' in names_new:
		names_new.remove('t')

	# plot
	T : NDArray = data_raw['t']
	V : Dict[str, NDArray] = {
		x : data_raw[x]
		for x in names_new
	}

	for v_name,v_arr in V.items():
		# print(v_name, v_arr.shape, v_arr.dtype)
		plt.plot(T, v_arr, label = v_name)

	plt.title(rootdir)
	plt.xlabel('time')
	plt.ylabel('neuron output')
	
	plt.legend()
	if show:
		plt.show()


if __name__ == '__main__':
	import fire # type: ignore
	fire.Fire(plot_act)