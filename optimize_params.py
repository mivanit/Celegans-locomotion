"""
# genetic algorithm notes
- look into simulated annealing
- make a "lean" version with:
	- no printing of position/activation @ every timestep, only print at end
	- dont print anything except the final position to stdout
"""

from typing import *
import json

from pyutil.util import Path, joinPath,mkdir
from pyutil.genetic_utils import (
	run_genetic_algorithm,
	continue_genetic_algorithm,
)


def run_genetic_algorithm_loadJSON(cfgfile : Path):

	raise NotImplementedError("this function wont work yet due to some parameters being callables")

	# get the specified json file
	with open(cfgfile, 'r') as f_json:
		config : Dict[str,Any] = json.load(f_json)
	
	# copy the read-in contents to the run's folder
	if "rootdir" not in config:
		raise KeyError("missing 'rootdir' key!")

	mkdir(config["rootdir"])
	with open(joinPath(config["rootdir"], 'run_config.json'), 'w') as f_out:
		json.dump(config, f_out)

	# run the main function, passing params
	run_genetic_algorithm(**config)


if __name__ == '__main__':
	import fire # type: ignore
	fire.Fire({
		'run' : run_genetic_algorithm,
		# 'run_json' : run_genetic_algorithm_loadJSON,
		'continue' : continue_genetic_algorithm,
	})
















