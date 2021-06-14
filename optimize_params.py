"""
# genetic algorithm notes
- look into simulated annealing
- make a "lean" version with:
	- no printing of position/activation @ every timestep, only print at end
	- dont print anything except the final position to stdout
"""

from typing import *

from pyutil.genetic_utils import run_genetic_algorithm

if __name__ == '__main__':
	import fire # type: ignore
	fire.Fire(run_genetic_algorithm)
















