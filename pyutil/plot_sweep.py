from typing import *
import re
import os

import numpy as np
from nptyping import NDArray

import matplotlib.pyplot as plt

from util import Path,joinPath
from plot_pos import read_body_data,CoordsArr,CoordsRotArr



def get_headpos(data : NDArray[(Any,Any), CoordsRotArr]):
	return np.array([ data[0]['x'], data[0]['y'] ])


def compare_dist_foodLR(
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
	lst_wgt_dirs : List[Path] = list(filter(lambda p : os.path.isdir(joinPath(rootdir, p)), lst_wgt_dirs))
	
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
	arr_wgt,arr_dist = zip(*sorted(dct_weight_to_dist.items()))
	plt.plot(arr_wgt, arr_dist, 'bo')

	if show:
		plt.show()



if __name__ == '__main__':
	import fire
	fire.Fire(compare_dist_foodLR)