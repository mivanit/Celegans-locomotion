from typing import *
import re
import os,sys

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore

import matplotlib.pyplot as plt # type: ignore


__EXPECTED_PATH__ : str = 'pyutil.plot.sweep'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import Path,joinPath,split_dict_arrs
from pyutil.plot.pos import read_body_data,CoordsArr,CoordsRotArr



def get_headpos(data : NDArray[(Any,Any), CoordsRotArr]):
	return np.array([ data[0]['x'], data[0]['y'] ])

class SweepPlotters(object):
	@staticmethod
	def compare_foodLR_dist(
			rootdir : Path = 'data/run/',
			bodydat : Path = 'body.dat',
			# params : Path = 'params.json',
			idx : Optional[int] = None,
			to_compare : Tuple[str,str] = ('food_left','food_right'),
			show : bool = True,
		):
		# we will put the data in here
		dct_weight_to_dist : Dict[float,float] = dict()

		# get all the directories, loop over them
		lst_wgt_dirs : List[Path] = os.listdir(rootdir)
		# filter out only the directories
		lst_wgt_dirs = list(filter(lambda p : os.path.isdir(joinPath(rootdir, p)), lst_wgt_dirs))
		
		count : int = 1
		count_max : int = len(lst_wgt_dirs)

		for wgt_dir in lst_wgt_dirs:
			# figure out the weight
			wgt : float = float(wgt_dir.split('_')[-1])
			print(f'  >>  loading data for weight = {wgt} \t ({count} / {count_max})')
			
			# get data for both sides
			data_L : NDArray[(Any,Any), CoordsRotArr] = read_body_data(joinPath(rootdir,wgt_dir,to_compare[0],bodydat))
			data_R : NDArray[(Any,Any), CoordsRotArr] = read_body_data(joinPath(rootdir,wgt_dir,to_compare[1],bodydat))
			
			# get the index -- this only happens once, if at all
			if idx is None:
				idx = data_L.shape[0] - 1

			# store distance
			dct_weight_to_dist[wgt] = np.linalg.norm(
				get_headpos(data_L[idx]) - get_headpos(data_R[idx]),
				ord = 2,
			)

			count += 1

		# plot
		arr_wgt,arr_dist = split_dict_arrs(dct_weight_to_dist)
		plt.plot(arr_wgt, arr_dist, 'bo')

		plt.xlabel('connection strength')
		plt.ylabel('L2 distance between final head positions for food left, food right')
		plt.title(rootdir)

		if show:
			plt.show()

	@staticmethod
	def compare_foodLR_xpos(
			rootdir : Path = 'data/run/',
			bodydat : Path = 'body.dat',
			# params : Path = 'params.json',
			idx : Optional[int] = None,
			to_compare : Tuple[str,str] = ('food_left','food_right'),
			show : bool = True,
		):
		# we will put the data in here
		dct_weight_to_xpos_foodL : Dict[float,float] = dict()
		dct_weight_to_xpos_foodR : Dict[float,float] = dict()

		# get all the directories, loop over them
		lst_wgt_dirs : List[Path] = os.listdir(rootdir)
		# filter out only the directories
		lst_wgt_dirs = list(filter(lambda p : os.path.isdir(joinPath(rootdir, p)), lst_wgt_dirs))
		
		count : int = 1
		count_max : int = len(lst_wgt_dirs)

		for wgt_dir in lst_wgt_dirs:
			# figure out the weight
			wgt : float = float(wgt_dir.split('_')[-1])
			print(f'  >>  loading data for weight = {wgt} \t ({count} / {count_max})')
			
			# get data for both sides
			data_L : NDArray[(Any,Any), CoordsRotArr] = read_body_data(joinPath(rootdir,wgt_dir,to_compare[0],bodydat))
			data_R : NDArray[(Any,Any), CoordsRotArr] = read_body_data(joinPath(rootdir,wgt_dir,to_compare[1],bodydat))
			
			# get the index -- this only happens once, if at all
			if idx is None:
				idx = data_L.shape[0] - 1

			# store distance
			dct_weight_to_xpos_foodL[wgt] = data_L[idx][0]['x']
			dct_weight_to_xpos_foodR[wgt] = data_R[idx][0]['x']

			count += 1

		# plot
		plt.plot(*split_dict_arrs(dct_weight_to_xpos_foodL), 'o', label = 'food left')
		plt.plot(*split_dict_arrs(dct_weight_to_xpos_foodR), 'o', label = 'food right')
		plt.xlabel('connection strength')
		plt.ylabel('x-axis position at end of run')
		plt.title(rootdir)

		plt.legend()

		if show:
			plt.show()


if __name__ == '__main__':
	import fire # type: ignore
	fire.Fire(SweepPlotters)