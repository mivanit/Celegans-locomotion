from typing import *
import os
import sys
import glob
from collections import defaultdict

import numpy as np # type: ignore
import numpy.lib.recfunctions as rfn # type: ignore
from nptyping import NDArray,StructuredType # type: ignore

import matplotlib.pyplot as plt # type: ignore

if TYPE_CHECKING:
	from pyutil.util import (
		Path,joinPath,unixPath,
		ParamsDict,ModParam,ModParamsDict,ModParamsRanges,
		wrapper_printdict,
	)
else:
	from util import (
		Path,joinPath,unixPath,
		ParamsDict,ModParam,ModParamsDict,ModParamsRanges,
		wrapper_printdict,
	)

EXTRACTED_FILENAME : Path = "extracted.txt"

def scrape_extracted(
		rootdir : Path,
		cast_func : Callable[[str], Any] = float,
		comment_str : str = '#',
		top_n : Optional[int] = None,
	) -> Dict[Path,str]:

	map_extracted : Dict[Path,str] = dict()

	lst_data : List[Path] = glob.glob(
		joinPath(unixPath(rootdir),f"**/{EXTRACTED_FILENAME}"), 
		recursive = True,
	)

	for p in lst_data:
		
		p_trim : Path = unixPath(
			unixPath(p)
			.replace(rootdir, '')
			.replace(EXTRACTED_FILENAME, '')
		)

		with open(p, 'r') as fin:
			map_extracted[p_trim] = cast_func('\n'.join([
				x
				for x in fin.readlines()
				if not x.startswith(comment_str)
			]))

	# sort it all
	map_extracted = {
		k[0] : k[1]
		for k in sorted(
			map_extracted.items(), 
			key = lambda x : x[1], 
			reverse = True,
		)[:top_n]
	}
	
	return map_extracted


def generational_histogram(
		rootdir : Path, 
		n_bins : Optional[int] = 20,
		show : bool = True,
	):
	data : Dict[Path,float] = scrape_extracted(rootdir)

	# first make bins based on all data
	bins : NDArray = np.linspace(
		min(data.values()), 
		max(data.values()), 
		num = n_bins + 1,
		endpoint = True,
	)

	bin_centers : NDArray = 0.5 * (bins[1:] + bins[:-1])

	# sort by generation
	sorted_data : DefaultDict[Path, List[float]] = defaultdict(list)
	for k,v in data.items():
		gen_k : Path = k.split('/')[0]
		sorted_data[gen_k].append(v)
	

	# plot each generation
	for gen_k,lst_v in sorted_data.items():		
		hist,_ = np.histogram(lst_v, bins)
		plt.plot(
			bin_centers, hist,
			'o-',
			label = gen_k, 
		)
	
	plt.legend()
	
	if show:
		plt.show()


if __name__ == '__main__':
	import fire
	data = fire.Fire({
		"list" : wrapper_printdict(scrape_extracted),
		"hist" : generational_histogram,
	})

