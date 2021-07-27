from typing import *
import os
import sys
import glob
from collections import defaultdict

import numpy as np # type: ignore
import numpy.lib.recfunctions as rfn # type: ignore
from nptyping import NDArray,StructuredType # type: ignore

import matplotlib.pyplot as plt # type: ignore
from matplotlib import cm # type: ignore

__EXPECTED_PATH__ : str = 'pyutil.plot.gene'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import *
from pyutil.params import *

"""
from pyutil.util import (
	Path,joinPath,unixPath,GeneRunID,
	wrapper_printdict,raise_,
)

from pyutil.params import (
	ParamsDict,ModParam,
	ModParamsDict,ModParamsRanges,
)
"""

CACHE_FILE : Path = "extracted_cache"
EXTRACTED_FILENAME : Path = "extracted.txt"

def scrape_extracted_old(
		rootdir : Path,
		cast_func : Callable[[str], Any] = float,
		comment_str : str = '#',
		n_top : Optional[int] = None,
	) -> Dict[Path,str]:

	map_extracted : Dict[Path,str] = dict()

	lst_data : List[Path] = glob.glob(
		joinPath(unixPath(rootdir),f"**/{EXTRACTED_FILENAME}"), 
		recursive = True,
	)

	n_items : int = len(lst_data)

	print(f'> will read {n_items} items\n\n')

	for idx,p in enumerate(lst_data):

		if idx % 100 == 0:
			print(f"  >> read \t{idx}\t/\t{n_items}", end = "\r")
		
		p_trim : Path = unixPath(
			unixPath(p)
			.replace(rootdir, '')
			.replace(EXTRACTED_FILENAME, '')
		)

		with open(p, 'r') as fin:
			map_extracted[p_trim] = cast_func([
				x
				for x in fin.readlines()
				if not x.startswith(comment_str)
			][-1])

	# sort it all
	map_extracted = {
		k[0] : k[1]
		for k in sorted(
			map_extracted.items(), 
			key = lambda x : x[1], 
			reverse = True,
		)[:n_top]
	}
	
	return map_extracted

def extractedKeys_to_str(data : Dict[GeneRunID, float]) -> Dict[str,float]:
	return {
		repr(k) : v
		for k,v in data.items()
	}

def extractedKeys_to_GRI(data : Dict[str, float]) -> Dict[GeneRunID,float]:
	return {
		GeneRunID.from_str(k) : v
		for k,v in data.items()
	}


def scrape_extracted_cache(
		rootdir : Path,
		cast_func : Callable[[str], Any] = float,
		comment_str : str = '#',
		n_top : Optional[int] = None,
		use_cache : bool = True,
		format : Literal['json','msgpack'] = 'json',
	) -> Dict[GeneRunID, float]:

	cache_file : Path = joinPath(rootdir, CACHE_FILE)
	# UGLY: this bit
	cache_load : Callable[[],Dict[str,Any]] = lambda : raise_(NotImplementedError('check `scrape_extracted_cache()`'))
	cache_save : Callable[[Dict[str,float]],None] = lambda x : raise_(NotImplementedError('check `scrape_extracted_cache()`'))
	if format == 'msgpack':
		import msgpack # type: ignore
		cache_file += '.mpk'
		cache_load = lambda : msgpack.load(open(cache_file,'rb'))
		cache_save = lambda x : msgpack.pack(x,open(cache_file,'wb'))
	elif format == 'json':
		import json
		cache_file += '.json'
		cache_load = lambda : json.load(open(cache_file,'rt'))
		cache_save = lambda x : json.dump(x,open(cache_file,'wt'), indent='\t')
	else:
		raise KeyError(f'unknown format {format}')
	
	map_extracted : Dict[GeneRunID, float] = dict()
	
	did_read_cache : bool = False
	if use_cache and os.path.isfile(cache_file):
			map_extracted = extractedKeys_to_GRI(cache_load())
			print(f'> got {len(map_extracted)} items from cache at {cache_file}\n\n')
			did_read_cache = True
	else:

		lst_data : List[Path] = glob.glob(
			joinPath(unixPath(rootdir),f"**/{EXTRACTED_FILENAME}"), 
			recursive = True,
		)

		n_items : int = len(lst_data)

		print(f'> will read {n_items} items\n\n')

		for idx,p in enumerate(lst_data):

			if (idx % 100 == 0) or (idx == n_items - 1):
				print(f"  >> read \t{idx+1}\t/\t{n_items}", end = "\r")
			
			p_trim : Path = unixPath(
				unixPath(p)
					.replace(rootdir, '')
					.replace(EXTRACTED_FILENAME, '')
					.strip(' /')
			)

			p_tup : GeneRunID = GeneRunID.from_str(p_trim)

			with open(p, 'r') as fin:
				map_extracted[p_tup] = cast_func([
					x
					for x in fin.readlines()
					if not x.startswith(comment_str)
				][-1])
		
		print('\n')

	if not did_read_cache:
		print(f'> writing {len(map_extracted)} items to cache at {cache_file}\n\n')
		cache_save(extractedKeys_to_str(map_extracted))

	# sort it all
	map_extracted = {
		k[0] : k[1]
		for k in sorted(
			map_extracted.items(), 
			key = lambda x : x[1], 
			reverse = True,
		)[:n_top]
	}

	
	return map_extracted


def get_bins(data : Dict[GeneRunID, float], n_bins : int = 20) -> Tuple[NDArray,NDArray]:
	bins : NDArray = np.linspace(
		min(data.values()), 
		max(data.values()), 
		num = n_bins + 1,
		endpoint = True,
	)

	bin_centers : NDArray = 0.5 * (bins[1:] + bins[:-1])

	return bins,bin_centers

def sort_by_generation(data : Dict[GeneRunID, float]) -> DefaultDict[int, List[float]]:
	data_sorted : DefaultDict[int, List[float]] = defaultdict(list)
	for k,v in data.items():
		data_sorted[k.gen].append(v)
	
	return data_sorted

def generational_histogram(
		rootdir : Path, 
		n_bins : int = 20,
		min_gen : int = 0,
		show : bool = True,
	):
	# OPTIMIZE: only scrape the ones with gen > min_gen
	data : Dict[GeneRunID, float] = scrape_extracted_cache(rootdir)

	# first make bins based on all data
	bins,bin_centers = get_bins(data, n_bins)

	# sort by generation
	data_sorted : DefaultDict[int, List[float]] = sort_by_generation(data)

	# plot each generation
	gen_count : int = max(data_sorted.keys())
	colors = cm.plasma(np.linspace(0,1,gen_count-min_gen)) # type: ignore

	for gen_k in range(min_gen,gen_count):
		lst_v : List[float] = data_sorted[gen_k]
		
		hist,_ = np.histogram(lst_v, bins)
		plt.plot(
			bin_centers, hist,
			'-',
			label = f"gen_{gen_k}", 
			color = colors[gen_k-min_gen],
		)
	
	plt.legend()
	
	if show:
		plt.show()


if __name__ == '__main__':
	import fire # type: ignore
	data = fire.Fire({
		"list" : wrapper_printdict(scrape_extracted_cache),
		"hist" : generational_histogram,
	})

