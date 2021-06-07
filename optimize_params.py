"""
# genetic algorithm notes
- look into simulated annealing
- make a "lean" version with:
	- no printing of position/activation @ every timestep, only print at end
	- dont print anything except the final position to stdout
"""

from typing import *
import subprocess
import copy
from math import dist
import random
import json

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore
from pydbg import dbg

if TYPE_CHECKING:
	from mypy_extensions import Arg
else:
	Arg = lambda t,s : t

from pyutil.util import (
	ModParam, ModTypes, Path,mkdir,joinPath,
	strList_to_dict,ParamsDict,ModParamsDict,ModParamsRanges,
	MODPARAMS_DEFAULT_RANGES,
	VecXY,dump_state,
	find_conn_idx,find_conn_idx_regex,
	genCmd_singlerun,
	dict_hash,load_params,
	keylist_access_nested_dict,
	read_body_data,CoordsRotArr,
	prntmsg,
)

Process = Any
Population = List[ModParamsDict]
PopulationFitness = List[
	Tuple[ModParamsDict, Optional[float]]
]



"""

 #    # ###### #####   ####  ######
 ##  ## #      #    # #    # #
 # ## # #####  #    # #      #####
 #    # #      #####  #  ### #
 #    # #      #   #  #    # #
 #    # ###### #    #  ####  ######

"""

def merge_params_with_mods(
		# base params
		params_base : ParamsDict,
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
	) -> ParamsDict:
	"""merges a params file with a special "mod" dict
	
	returns a modified copy of `params_base`, modified according to the contents of `params_mod`
	`params_base` is of the same form as a regular params.json file,
	but `params_mod` has the following structure:

	```python
	params_mod = {
			('params','Head.neurons.AWA.theta' : 2.0,
			('params','ChemoReceptors.alpha' : 2.0,
			('conn','Head,AWA,RIM,chem' : 10.0,
		}
	}
	```

	- keys starting with `params` map dot-separated keys to the nested params dict, to their desired values
	- keys starting with `conn` map comma-separated connection identifiers to their desired values
	
	### Parameters:
	 - `params_mod : ModParamsDict` 
	   special dict to modify a copy of `params_base`
	 - `params_base : ParamsDict`
	   `params.json` style dict
	
	### Returns:
	 - `ParamsDict` 
	   modified copy of `params_base`
	"""

	# copy the input dict
	output : dict = copy.deepcopy(params_base)

	# REVIEW: why did i even refactor this when im making everything editable through params json anyway?
	for tup_key,val in params_mod.items():
		# merge in the standard params
		if tup_key.mod_type == ModTypes.params.value:
			
			nested_keys : str = tup_key.path

			fin_dic,fin_key = keylist_access_nested_dict(
				d = output, 
				keys = nested_keys.split('.'),
			)
			fin_dic[fin_key] = val

		elif tup_key.mod_type == ModTypes.conn.value:
			# merge in the connection modifiers
			conn_key_str : str = tup_key.path
			conn_key = strList_to_dict(
				in_data = conn_key_str,
				keys_list = ['NS', 'from', 'to', 'type'],
				delim = ',',
			)

			# get the indecies of the connections whose weights need to be changed
			conn_idxs : List[Optional[int]] = find_conn_idx_regex(
				params_data = output, 
				conn_key = conn_key,
			)

			# set weights
			for cidx in conn_idxs:
				output[conn_key['NS']]['connections'][cidx]['weight'] = val
		else:
			raise NotImplementedError(f'given key type {tup_key.mod_type} unknown')

	return output


"""

 ###### #    # ##### #####    ##    ####  #####
 #       #  #    #   #    #  #  #  #    #   #
 #####    ##     #   #    # #    # #        #
 #        ##     #   #####  ###### #        #
 #       #  #    #   #   #  #    # #    #   #
 ###### #    #   #   #    # #    #  ####    #

"""

ExtractorReturnType = Any

ExtractorFunc = Callable[
	[
		Arg(Path, 'datadir'),
		Arg(ParamsDict, 'params'),
		Arg(bool, 'ret_nan'),
	], 
	ExtractorReturnType, # return type
]



def _wrapper_extract(
		proc, 
		func_extract : ExtractorFunc, 
		outpath : Path, 
		params_joined : ParamsDict,
	):	
	# wait for command to finish
	proc.wait()
		
	if proc.returncode:
		print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n        {str(proc.args)}')

	return func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(proc.returncode),
	)



def _extract_TEMPLATE(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> ExtractorReturnType:
	"""template function for extraction functions
	
	dont actually call this function. it contains documentation for the format of functions 
	`func_extract` taken by `evaluate_params()`
	
	### Parameters:
	 - `datadir : Path`   
	   output directory of data
	 - `params : ParamsDict`   
	   nested dictionary of params
	 - `ret_nan : bool`   
	   whether to return nan value (when process terminates in error)
	   (defaults to `False`)
	
	### Returns:
	 - `ExtractorReturnType` 
	   can return any data about the run
	
	### Raises:
	 - `NotImplementedError` : dont run this!
	"""
	
	raise NotImplementedError('this is a template function only!')


def extract_finalpos(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> Tuple[float,float]:
	"""extract just the final head position
	
	### Returns:
	 - `Tuple[float,float]` 
	   head position
	"""
	if ret_nan:
		return ( float('nan'), float('nan') )	
	else: 
		bodydata : CoordsRotArr = read_body_data(datadir + 'body.dat')[-1,0]
		return ( bodydata['x'], bodydata['y'] )

def extract_food_dist(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> float:
	"""extract euclidean distance from head to food
	
	### Returns:
	 - `float` 
	   dist from final head position to food
	"""
	if ret_nan:
		return float('nan')
	else:
		# get head pos
		bodydata : CoordsRotArr = read_body_data(datadir + 'body.dat')[-1,0]
		pos_head : VecXY = VecXY( bodydata['x'], bodydata['y'] )

		# get food pos
		pos_food : VecXY = VecXY(
			params['ChemoReceptors']['foodPos']['x'],
			params['ChemoReceptors']['foodPos']['y'],
		)

		# return distance
		return dist(pos_head, pos_food)

def extract_food_dist_inv(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> float:
	"""extract inverse of euclidean distance from head to food

	this means that higher value ==> higher fitness
	
	### Returns:
	 - `float` 
	   1/(dist from final head position to food)
	"""
	return 1 / extract_food_dist(datadir, params, ret_nan)


def extract_df_row(
		datadir : Path,
		params : ParamsDict,
		ret_nan : bool = False,
	) -> dict:
	# TODO: implement extracting more data, for parameter sweeps
	raise NotImplementedError('please implement me :(')


"""

 ###### #    #   ##   #
 #      #    #  #  #  #
 #####  #    # #    # #
 #      #    # ###### #
 #       #  #  #    # #
 ######   ##   #    # ######

"""

def setup_evaluate_params(
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
		# base params
		params_base : ParamsDict,
		# root directory for run
		rootdir : Path = 'data/run/anneal/',
		# extract info from the final product
		func_extract : ExtractorFunc = extract_food_dist,
		# command line args
		rand : Optional[bool] = None,
	) -> Tuple[Process, Path, ParamsDict]:
	# TODO: document this
	
	# make dir
	outpath : Path = f"{rootdir}r-{dict_hash(params_mod)}/"
	outpath_params : Path = joinPath(outpath,'in-params.json')
	mkdir(outpath)

	# join params
	params_joined : ParamsDict = merge_params_with_mods(params_base, params_mod)

	# save modified params
	with open(outpath_params, 'w') as fout:
		json.dump(params_joined, fout, indent = '\t')

	# set up the command by passing kwargs down
	cmd : str = genCmd_singlerun(
		params = outpath_params,
		output = outpath,
		# **kwargs,
	)

	# run the process, write stderr and stdout to the log file
	with open(outpath + 'log.txt', 'w') as f_log:
		proc = subprocess.Popen(
			cmd, 
			stderr = subprocess.STDOUT,
			stdout = f_log,
		)

	return (proc, outpath, params_joined)
	
def evaluate_params(
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
		# base params
		params_base : ParamsDict,
		# root directory for run
		rootdir : Path = 'data/run/anneal/',
		# extract info from the final product
		func_extract : ExtractorFunc = extract_food_dist,
		# command line args
		rand : Optional[bool] = None,
	) -> ExtractorReturnType:
	
	proc, outpath, params_joined = setup_evaluate_params(
		params_mod = params_mod,
		params_base = params_base,
		rootdir = rootdir,
	)

	# wait for command to finish
	proc.wait()
		
	if proc.returncode:
		print(f'  >>  ERROR: process terminated with exit code 1, check log.txt for:\n        {str(proc.args)}')

	return func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(proc.returncode),
	)


"""
 ######   ######## ##    ## ########
##    ##  ##       ###   ## ##
##        ##       ####  ## ##
##   #### ######   ## ## ## ######
##    ##  ##       ##  #### ##
##    ##  ##       ##   ### ##
 ######   ######## ##    ## ########
"""


"""

 #    # #    # #####
 ##  ## #    #   #
 # ## # #    #   #
 #    # #    #   #
 #    # #    #   #
 #    #  ####    #

"""

def mutate_state(
		params_mod : ModParamsDict,
		ranges : ModParamsRanges,
		mutprob : float = 0.01,
		sigma : float = 0.1,
	) -> None:
	
	# choose a variable to mutate
	choice_key : ModParam = random.choice(list(params_mod.keys()))
	choice_val : float = params_mod[choice_key]

	# modify the value according to the range
	delta_val : float = random.gauss(0, sigma)
	params_mod[choice_key] = choice_val + delta_val


# TODO: review this function type
GenoCombineFunc = Callable
# GenoCombineFunc = Callable[
# 	[ModParamsDict, ModParamsDict],
# 	ModParamsDict,
# ]


"""

  ####  #####   ####   ####   ####
 #    # #    # #    # #      #
 #      #    # #    #  ####   ####
 #      #####  #    #      #      #
 #    # #   #  #    # #    # #    #
  ####  #    #  ####   ####   ####

"""

def combine_geno_select(
		pmod_A : ModParamsDict,
		pmod_B : ModParamsDict,
	) -> ModParamsDict:
	
	# assert that keys match
	assert all(
		(k in pmod_B) 
		for k in pmod_A.keys()
	), 'keys dont match!'
	
	choices : List[bool] = list(np.random.choice([True,False], len(pmod_A)))
	output : ModParamsDict = dict()

	for key in pmod_A:
		c : bool = choices.pop()
		if c:
			output[key] = pmod_A[key]
		else:
			output[key] = pmod_B[key]
	
	return output

def combine_geno_mean_normal(
		pmod_A : ModParamsDict,
		pmod_B : ModParamsDict,
		noise_sigma : float = 0.1,
		threshold_noise : float = 0.00001,
		# ranges : Optional[ModParamsRanges] = None,
	) -> ModParamsDict:
	"""combines `pmod_A, pmod_B` into a single genotype
	
	when modifying an individual component of the genotypes,
	if the difference is over `threshold_noise`,
	then we take the average between the values 
	and add noise using `noise_sigma` as sigma for normal distribution
	if difference < `threshold_noise`, we just take the average
	
	notes:
	- keys of `pmod_A, pmod_B` should match
	- order of `pmod_A, pmod_B` shouldnt matter

	### Parameters:
	 - `pmod_A : ModParamsDict`   
	   dict of parameters being modified
	 - `pmod_B : ModParamsDict`   
	   dict of parameters being modified
	 - `noise_sigma : float`   
	   noise added to average of every parameter is given by `noise_sigma * val_range`
	   (defaults to `0.1`)
	 - `threshold_noise : float`   
	   if difference is less than this value, dont add noise (assume they are equal)
	   (defaults to `0.00001`)
	
	### Returns:
	 - `ModParamsDict` 
	   [description]
	"""
	pmod_out : ModParamsDict = dict()

	# assert that keys match
	assert all(
		(k in pmod_B) 
		for k in pmod_A.keys()
	), 'keys dont match!'

	for key in pmod_A.keys():
		# set new val to average
		val : float = (pmod_A[key] + pmod_B[key]) / 2.0
		
		# add noise if difference is big enough
		val_range : float = abs(pmod_A[key] - pmod_B[key])
		if val_range > threshold_noise:
			val += random.gauss(0.0, val_range * noise_sigma)

		# store new val
		pmod_out[key] = val

	return pmod_out




def generation_reproduction(
		pop : PopulationFitness,
		popsize_new : int,
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
		# chance_direct_progression : float = 0.2,
	) -> PopulationFitness:

	popsize_old : int = len(pop)
	newpop : PopulationFitness = list()

	dbg(len(pop))
	dbg(popsize_old)
	dbg(popsize_new)
	random_selection : NDArray = np.random.randint(
		low = 0, 
		high = popsize_old, 
		size = (popsize_new, 2),
	)

	n_indiv : int = 0
	while len(newpop) < popsize_new:
		
		prm_A : ModParamsDict = pop[random_selection[n_indiv][0]][0]
		prm_B : ModParamsDict = pop[random_selection[n_indiv][1]][0]
		prm_comb : ModParamsDict = gene_combine(prm_A, prm_B, **gene_combine_kwargs)
	
		newpop.append((prm_comb, None))

		n_indiv += 1
	
	return newpop


"""

  ####  ###### #    #
 #    # #      ##   #
 #      #####  # #  #
 #  ### #      #  # #
 #    # #      #   ##
  ####  ###### #    #

"""

def generate_geno_uniform(
		ranges : ModParamsRanges,
	) -> ModParamsDict:

	raise NotImplementedError()

def generate_geno_uniform_many(
		ranges : ModParamsRanges,
		n_genos : int,
	) -> PopulationFitness:

	random_vals : Dict[ModParam, NDArray[n_genos, float]] = {
		pr : np.random.uniform(rn.min, rn.max, size = n_genos)
		for pr,rn in ranges.items()
	}

	return [
		(
			{
				pr : random_vals[pr][i]
				for pr in ranges.keys()
			},
			None,
		)	
		for i in range(n_genos)
	]
	
	


"""

 ###### #    #   ##   #
 #      #    #  #  #  #
 #####  #    # #    # #
 #      #    # ###### #
 #       #  #  #    # #
 ######   ##   #    # ######

"""

def eval_pop_fitness(
		rootdir : Path,
		params_base : ParamsDict,
		pop : PopulationFitness,
		extractorfunc : ExtractorFunc,
	) -> PopulationFitness:

	prntmsg(f'evaluating fitness of population of size {len(pop)}, storing in {rootdir}', 2)

	# a mapping of parameters to fitness
	output_fitness : PopulationFitness = list()
	
	# a list of processes that we instantiate,
	# whose results need to be added to `output_fitness` once they terminate
	to_read : List[Tuple[ParamsDict, ModParamsDict, Process, Path]] = list()

	# start all the required processes
	for prm_mod,fit in pop:
		if fit is None:
			proc, outpath, prm_join = setup_evaluate_params(
				params_mod = prm_mod,
				params_base = params_base,
				rootdir = rootdir,
			)
			to_read.append((prm_join, prm_mod, proc, outpath))
		else:
			# if fitness is known, dont recalculate
			output_fitness.append((prm_mod, fit))

	# wait for them to finish, then read fitness
	for prm_join,prm_mod,proc,outpath in to_read:
		proc.wait()

		new_fit : float = extractorfunc(
			datadir = outpath,
			params = prm_join,
			ret_nan = proc.returncode,
		)

		output_fitness.append((prm_mod, new_fit))

	# return the results
	return output_fitness

"""

  ####  ###### #      ######  ####  #####
 #      #      #      #      #    #   #
  ####  #####  #      #####  #        #
      # #      #      #      #        #
 #    # #      #      #      #    #   #
  ####  ###### ###### ######  ####    #

"""

def generation_selection(
		pop : PopulationFitness,
		new_popsize : int,
		# prob_allow_lessfit : float = 0.1,
	) -> PopulationFitness:
	"""
	select a number of individals to survive to the next generation
	"""
	# fitness : PopulationFitness = eval_pop_fitness(pop, extractorfunc)

	if new_popsize == len(pop):
		return copy.deepcopy(pop)

	# TODO: fix typing here
	assert not any(
		f is None
		for _,f in pop
	), "`None` fitness found when trying to run `generation_selection`"

	lst_fit : List[float] = sorted((f for _,f in pop), reverse = True) # type: ignore
	fitness_thresh : float = lst_fit[new_popsize]

	newpop : PopulationFitness = [
		(prm,fit)
		for prm,fit in pop
		if (fit > fitness_thresh) # type: ignore
	]

	# TODO: pop/push if the element count is not quite right?

	return newpop
	





"""

 #####  #    # #    #
 #    # #    # ##   #
 #    # #    # # #  #
 #####  #    # #  # #
 #   #  #    # #   ##
 #    #  ####  #    #

"""

def run_generation(
		pop : PopulationFitness,
		rootdir : Path,
		params_base : ParamsDict,
		popsize_select : int,
		popsize_new : int,
		# ranges : ModParamsRanges,
		# sigma : float,
		extractorfunc : ExtractorFunc,
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
	) -> PopulationFitness:

	dbg(len(pop))

	# trim old population
	pop_trimmed : PopulationFitness = generation_selection(pop, popsize_select)

	dbg(len(pop_trimmed))

	# run reproduction
	pop_new : PopulationFitness = generation_reproduction(
		pop = pop_trimmed,
		popsize_new = popsize_new,
		gene_combine = gene_combine,
		gene_combine_kwargs = gene_combine_kwargs,
	)

	# mutate
	# TODO: implement mutation

	dbg(len(pop_new))

	# evaluate fitness of new individuals
	return eval_pop_fitness(
		rootdir = rootdir,
		params_base = params_base,
		pop = pop_new,
		extractorfunc = extractorfunc, 
	)



"""

 #####  #    # #    #
 #    # #    # ##   #
 #    # #    # # #  #
 #####  #    # #  # #
 #   #  #    # #   ##
 #    #  ####  #    #

"""

def compute_gen_sizes(
		first_gen_size : int,
		gen_count : int,
		factor_cull : float,
		factor_repro : float,
	) -> List[Tuple[int,int]]:

	output : List[Tuple[int,int]] = [(first_gen_size, first_gen_size)]

	for g in range(gen_count):
		count_prev : int = output[g][1]
		count_cull : int = int(count_prev * factor_cull)
		count_new : int = int(count_cull * factor_repro)
		output.append((count_cull, count_new))
	
	return output



def run_genetic_algorithm(
		# for setup
		rootdir : Path = "data/geno_sweep/",
		ranges : ModParamsRanges = MODPARAMS_DEFAULT_RANGES,
		first_gen_size : int = 10,
		gen_count : int = 5,
		factor_cull : float = 0.45,
		factor_repro : float = 2.0,
		# passed to `run_generation`
		params_base : ParamsDict = load_params("input/chemo_v6.json"),
		# sigma : float = 0.1,
		extractorfunc : ExtractorFunc = extract_food_dist,
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
	) -> PopulationFitness:

	# compute population sizes
	pop_sizes : List[Tuple[int, int]] = compute_gen_sizes(
		first_gen_size = first_gen_size,
		gen_count = gen_count,
		factor_cull = factor_cull,
		factor_repro = factor_repro,
	)

	# generate initial population
	pop : PopulationFitness = generate_geno_uniform_many(
		ranges = ranges,
		n_genos = pop_sizes[0][1],
	)

	# run each generation
	for count_cull,count_new in pop_sizes:
		pop = run_generation(
			pop = pop,
			rootdir = rootdir,
			params_base = params_base,
			popsize_select = count_cull,
			popsize_new = count_new,
			# sigma = sigma,
			extractorfunc = extractorfunc,
			gene_combine = gene_combine,
			gene_combine_kwargs = gene_combine_kwargs,
		)

	# return final generation
	return pop

if __name__ == '__main__':
	import fire # type: ignore
	res = fire.Fire(run_genetic_algorithm)
	print(res)
















