from typing import *
import os
import sys
import glob

import numpy as np # type: ignore
import numpy.lib.recfunctions as rfn # type: ignore
from nptyping import NDArray,StructuredType # type: ignore

import matplotlib.pyplot as plt # type: ignore



if TYPE_CHECKING:
	from pyutil.util import (
		Path,joinPath,
		ParamsDict,ModParam,ModParamsDict,ModParamsRanges,
	)
else:
	from util import (
		Path,joinPath,
		ParamsDict,ModParam,ModParamsDict,ModParamsRanges,
	)

def scrape_extracted(
		rootdir : Path,
		cast_func : Callable[[str], Any] = float,
		comment_str : str = '#',
	) -> Dict[Path,str]:

	map_extracted : Dict[Path,str] = dict()

	for p in os.listdir(rootdir):
		ext_path : Path = joinPath(rootdir,p,"extracted.txt")
		
		with open(ext_path, 'r') as fin:
			map_extracted[p] = cast_func('\n'.join([
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
		)
	}
	
	return map_extracted


if __name__ == '__main__':
	import fire
	fire.Fire(scrape_extracted)

