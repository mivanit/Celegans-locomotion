from genericpath import isdir
from typing import *
import subprocess
import copy
import os
from math import dist,isnan
import random
import glob
import json

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore
from pydbg import dbg # type: ignore

__EXPECTED_PATH__ : str = 'pyutil.genetic_utils'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))


from pyutil.util import *
from pyutil.params import *
from pyutil.eval_run import *
from pyutil.geno_distr import DEFAULT_DISTS,DEFAULT_EVALRUNS

Process = Any
Population = List[ModParamsDict]
"""List of parameter dicts (genotypes)"""
PopulationFitness = List[
	Tuple[ModParamsDict, float]
]
"""Maps parameter dicts to fitness"""



"""
######## ##     ##    ###    ##
##       ##     ##   ## ##   ##
##       ##     ##  ##   ##  ##
######   ##     ## ##     ## ##
##        ##   ##  ######### ##
##         ## ##   ##     ## ##
########    ###    ##     ## ########
"""

def setup_evaluate_params(
		# modified params (this is what we are optimizing)
		params_mod : ModParamsDict,
		# base params
		params_base : ParamsDict,
		# root directory for run
		rootdir : Path,
		# extract info from the final product
		func_extract : ExtractorFunc = extract_food_dist,
		# command line args
		rand : Optional[bool] = None,
		out_name : Optional[Path] = None,
	) -> Tuple[Process, Path, ParamsDict]:
	# TODO: document this
	
	# make dir
	if out_name is None:
		outpath : Path = joinPath(rootdir, dict_hash(params_mod))
	else:
		outpath = joinPath(rootdir, out_name)
	
	if not outpath.endswith('/'):
		outpath = outpath + '/'
	
	mkdir(outpath)

	# join params
	params_joined : ParamsDict = merge_params_with_mods(params_base, params_mod)

	# save modified params
	outpath_params : Path = joinPath(outpath,'in-params.json')
	with open(outpath_params, 'w') as fout:
		json.dump(params_joined, fout, indent = '\t')

	# set up the command by passing kwargs down
	cmd : List[str] = genCmd_singlerun(
		params = outpath_params,
		output = outpath,
		# **kwargs,
	).split(' ')

	# run the process, write stderr and stdout to the log file
	with open(outpath + 'log.txt', 'a') as f_log:
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

	result : ExtractorReturnType = func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(proc.returncode),
	)

	with open(joinPath(outpath, 'extracted.txt'), 'a') as fout_ext:
		print(f'# extracted using {func_extract.__name__}:', file = fout_ext)
		print(repr(result), file = fout_ext)

	return result


"""
##     ## ##     ## ########    ###    ######## ########
###   ### ##     ##    ##      ## ##      ##    ##
#### #### ##     ##    ##     ##   ##     ##    ##
## ### ## ##     ##    ##    ##     ##    ##    ######
##     ## ##     ##    ##    #########    ##    ##
##     ## ##     ##    ##    ##     ##    ##    ##
##     ##  #######     ##    ##     ##    ##    ########
"""

def get_pop_ranges(pop : Population) -> ModParamsRanges:
	keys : List[ModParam] = list(pop[-1].keys())
	# OPTIMIZE: this uses two loops (because its just a view over the list) when it could be using just one
	output : ModParamsRanges = {
		key : RangeTuple(
			min(p[key] for p in pop),
			max(p[key] for p in pop),
		)
		for key in keys
	}

	return output

def mutate_state(
		params_mod : ModParamsDict,
		ranges : ModParamsRanges,
		mutprob : float,
		mut_sigma : float,
	) -> ModParamsDict:
	
	params_new : ModParamsDict = copy.deepcopy(params_mod)

	# each variable might be mutated
	for key,val in params_new.items():
		if random.random() < mutprob:
			delta_val : float = random.gauss(0, mut_sigma)
			params_new[key] = val + delta_val

	return params_new

# TODO: review this function type
GenoCombineFunc = Callable
# GenoCombineFunc = Callable[
# 	[ModParamsDict, ModParamsDict],
# 	ModParamsDict,
# ]
"""Typing for a function that combines geneotypes. needs review."""


"""
 ######  ########   #######   ######   ######
##    ## ##     ## ##     ## ##    ## ##    ##
##       ##     ## ##     ## ##       ##
##       ########  ##     ##  ######   ######
##       ##   ##   ##     ##       ##       ##
##    ## ##    ##  ##     ## ##    ## ##    ##
 ######  ##     ##  #######   ######   ######
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
		min_fitness : float = 0.0,
	) -> Population:

	popsize_old : int = len(pop)
	newpop : Population = list()

	# TODO: for some reason, `popsize_old` sometimes is less than or equal to zero. the max() is just a hack, since i dont know what causes the issue in the first place
	# random_selection : NDArray = np.random.randint(
	# 	low = 0, 
	# 	high = max(1,popsize_old), 
	# 	size = (popsize_new, 2),
	# )


	# choose `popsize_new` pairs of individuals, with probability weighted by their fitness
	random_selection : NDArray = np.random.choice(
		np.array([ key for key,_ in pop ]), 
		size = (popsize_new, 2), 
		p = norm_prob(np.array([
			fit - min_fitness
			if not isnan(fit)
			else min_fitness
			for key,fit in pop
		])),
	)

	for pair_params in random_selection:		
		prm_comb : ModParamsDict = gene_combine(
			pair_params[0], 
			pair_params[1], 
			**gene_combine_kwargs,
		)
	
		newpop.append(prm_comb)
	
	return newpop


"""

  ####  ###### #    #  ####           ####  ###### #    #
 #    # #      ##   # #    #         #    # #      ##   #
 #      #####  # #  # #    #         #      #####  # #  #
 #  ### #      #  # # #    #         #  ### #      #  # #
 #    # #      #   ## #    #         #    # #      #   ##
  ####  ###### #    #  ####           ####  ###### #    #
                             #######
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
			float('nan'),
		)	
		for i in range(n_genos)
	]

def generate_geno(
		dists : ModParamsDists,
		n_genos : int,
		ranges = None,
	) -> PopulationFitness:

	if ranges is not None:
		raise ValueError("This new code uses a new style of declaring initial parameter ranges, which allows for arbitrary initial distributions and not just normal distributions. since you're passing a `ranges` parameter, you are probably using old code and should be careful. if its just an old set of ranges, it should work fine though")

	# generate a dict mapping keys to lists of random values
	random_vals : Dict[ModParam, NDArray[n_genos, float]] = dict()
	
	# for each parameter, generate array depending on distribution
	for pr,dst in dists.items():
		if isinstance(dst, RangeTuple):
			random_vals[pr] = np.random.uniform(dst.min, dst.max, size = n_genos)
		elif isinstance(dst, NormalDistTuple):
			random_vals[pr] = np.random.normal(dst.mu, dst.sigma, size = n_genos)
		else:
			raise NotImplementedError(f"unknown distribution type:\t{pr}\t{dst}\t{type(dst)}")	

	# assemble
	return [
		(
			# this dict maps keys to the random values in the arrays
			{
				pr : random_vals[pr][i]
				for pr in dists.keys()
			},
			float('nan'), # this nan is the fitness, not a parameter value. quirk of how the populations are represented
		)	
		for i in range(n_genos)
	]
	
	


"""

 ###### #    #   ##   #      #####   ####  #####
 #      #    #  #  #  #      #    # #    # #    #
 #####  #    # #    # #      #    # #    # #    #
 #      #    # ###### #      #####  #    # #####
 #       #  #  #    # #      #      #    # #
 ######   ##   #    # ###### #       ####  #

"""

def eval_pop_fitness(
		pop : Population,
		rootdir : Path,
		params_base : ParamsDict,
		func_extract : ExtractorFunc,
		eval_runs : List[ModParamsDict] = [ dict() ],
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
		verbose : bool = False,
	) -> PopulationFitness:

	# wrap the extractor func for multiple runs
	func_extract_multi : MultiExtractorFunc = wrap_multi_extract(
		func_extract = func_extract,
		calc_mean = calcmean_symmetric,
	)

	# a mapping of parameters to fitness
	output_fitness : PopulationFitness = list()
	
	# a list of processes that we instantiate,
	# whose results need to be added to `output_fitness` once they terminate
	to_read : List[Tuple[
		ParamsDict,
		ModParamsDict,
		List[Process],
		Path,
	]] = list()

	# start all the required processes
	for prm_mod in pop:
		# for storing the running processes
		proc_temp : List[Process] = list()

		# create a folder
		outpath : Path = joinPath(rootdir, f"h{dict_hash(prm_mod)}")
		mkdir(outpath)

		# evaluate separately for every `eval_runs` option
		for er_v in eval_runs:
			
			# note that `er_v` takes priority
			proc, _, _ = setup_evaluate_params(
				params_mod = {**prm_mod, **er_v},
				params_base = params_base,
				rootdir = outpath,
				out_name = modprmdict_to_filename(er_v),
			)

			# store the process away, to be waited for later
			proc_temp.append(proc)
		
		# parameter dict, with parameters that are modified with `eval_runs` set to NaN
		params_join_naned : ParamsDict = merge_params_with_mods(
			merge_params_with_mods(params_base, prm_mod),
			jointo_nan_eval_runs(eval_runs)
		)

		# store everything away for later
		to_read.append((
			params_join_naned, 
			prm_mod, 
			proc_temp, 
			outpath,
		))

	# get the list of runs
	lst_ids : List[Path] = sorted([
		p.rstrip('/').split('/')[-1]
		for _,_,_,p in to_read
	])
	prntmsg(f'initialized {sum(len(x[2]) for x in to_read)} processes for {len(to_read)} individuals with unknown fitnesses', 2)
	if verbose:
		print(f'\n\t{" ".join(lst_ids)}\n')

	# wait for them to finish, then read fitness
	for prm_join,prm_mod,lst_proc,outpath in to_read:
		# OPTIMIZE: each process could be read as it finishes -- this is not perfectly efficient
		for p in lst_proc:
			p.wait()

		new_fit : float = func_extract_multi(
			datadir = outpath,
			params = prm_join,
			ret_nan = any(p.returncode != 0 for p in lst_proc),
		)

		# save extracted fitness to a file
		with open(joinPath(outpath, 'extracted.txt'), 'a') as fout_ext:
			print(f'# extracted using {func_extract_multi.__name__}:', file = fout_ext)
			print(repr(new_fit), file = fout_ext)

		# throw it in the list
		output_fitness.append((prm_mod, new_fit))

	# return the results
	return output_fitness


"""
 ######  ######## ##       ########  ######  ########
##    ## ##       ##       ##       ##    ##    ##
##       ##       ##       ##       ##          ##
 ######  ######   ##       ######   ##          ##
      ## ##       ##       ##       ##          ##
##    ## ##       ##       ##       ##    ##    ##
 ######  ######## ######## ########  ######     ##
"""

def fitness_distr(lst_fit : List[float]) -> Dict[str,float]:
	"""gets max, median, mean, and minimum fitness (assumes `lst_fit` is sorted)
	
	### Parameters:
	 - `lst_fit : List[Optional[float]]`
	"""
	# TODO: not the real median here, oops
	if lst_fit:
		return {
			'max' : lst_fit[0],
			'median' : lst_fit[len(lst_fit) // 2],
			'mean' : sum(lst_fit) / len(lst_fit),
			'min' : lst_fit[-1],
		}
	else:
		return {
			'max' : float('nan'),
			'median' : float('nan'),
			'mean' : float('nan'),
			'min' : float('nan'),
		}

def str_fitness_distr(lst_fit : List[float]) -> str:
	return ', '.join(
		f'{k} = {v:.6}'
		for k,v in fitness_distr(lst_fit).items()
	)
	

def generation_selection(
		pop : PopulationFitness,
		new_popsize : int,
		# prob_allow_lessfit : float = 0.1,
	) -> PopulationFitness:
	"""
	select a number of individals to survive to the next generation
	"""
	# fitness : PopulationFitness = eval_pop_fitness(pop, func_extract)

	if new_popsize == len(pop):
		return copy.deepcopy(pop)

	# TODO: fix typing here
	assert not any(
		f is None
		for _,f in pop
	), "`None` fitness found when trying to run `generation_selection`"

	lst_fit : List[float] = list(sorted((f for _,f in pop), reverse = True))
	fitness_thresh : float = lst_fit[new_popsize]

	prntmsg(f'fitness distribution: {str_fitness_distr(lst_fit)}', 2)
	prntmsg(f'trimming with fitness threshold approx {fitness_thresh}', 2)

	newpop : PopulationFitness = sorted(
		pop, 
		reverse = True, 
		key = lambda x : x[1],
	)[:new_popsize]
	# REEEEEEEEEEEEEEEEE i forgot a colon here

	# newpop : PopulationFitness = [
	# 	(prm,fit)
	# 	for prm,fit in pop
	# 	if (fit > fitness_thresh)
	# ]

	lst_fit_afterTrim : List[float] = sorted([fit for prm,fit in newpop], reverse=True)
	prntmsg(f'distribution after trim: {str_fitness_distr(lst_fit_afterTrim)}', 2)

	# TODO: pop/push if the element count is not quite right?

	return newpop
	




"""
 ######   ######## ##    ## ######## ########
##    ##  ##       ###   ## ##       ##     ##
##        ##       ####  ## ##       ##     ##
##   #### ######   ## ## ## ######   ########
##    ##  ##       ##  #### ##       ##   ##
##    ##  ##       ##   ### ##       ##    ##
 ######   ######## ##    ## ######## ##     ##
"""

def run_generation(
		pop : PopulationFitness,
		rootdir : Path,
		params_base : ParamsDict,
		popsize_select : int,
		popsize_new : int,
		mut_sigma : float,
		mutprob : float,
		func_extract : ExtractorFunc,
		eval_runs: List[ModParamsDict],
		ranges_override : Optional[ModParamsRanges] = None,
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
		n_gen : int = -1,
		verbose : bool = False,
	) -> PopulationFitness:

	# prntmsg(f' fitness of population of size {len(pop)}, storing in {rootdir}', 2)

	if n_gen == 0:
		return eval_pop_fitness(
			pop = [ x for x,_ in pop ],
			rootdir = rootdir,
			params_base = params_base,
			func_extract = func_extract, 
			eval_runs = eval_runs,
			calc_mean = calc_mean,
			verbose = verbose,
		)

	# UGLY: be able to modify the default fitness here
	min_fitness : float = min([
		fit
		if not isnan(fit)
		else 0.0
		for _,fit in pop 
	])

	# trim old population
	pop_trimmed : PopulationFitness = generation_selection(pop, popsize_select)

	# run reproduction
	pop_mated : Population = generation_reproduction(
		pop = pop_trimmed,
		popsize_new = popsize_new,
		gene_combine = gene_combine,
		gene_combine_kwargs = gene_combine_kwargs,
		min_fitness = min_fitness,
	)

	# mutate
	# we pass the ranges of the *current population*, otherwise sigma will be too big and cause huge mutations later on when the model begins to converge
	# REVIEW: I think this actually makes the mutations too small
	
	ranges : ModParamsRanges = dict()
	
	if ranges_override is None:
		ranges = get_pop_ranges([
			xa
			for xa,xb in pop_trimmed
		])
	else:
		ranges = ranges_override

	


	pop_mutated : Population = [
		mutate_state(
			params_mod = prm,
			ranges = ranges,
			mutprob = mutprob,
			mut_sigma = mut_sigma,
		)
		for prm in pop_mated
	]

	# evaluate fitness of new individuals
	return eval_pop_fitness(
		pop = pop_mutated,
		rootdir = rootdir,
		params_base = params_base,
		func_extract = func_extract, 
		eval_runs = eval_runs,
		calc_mean = calc_mean,
	)



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


"""

 #       ####    ##   #####
 #      #    #  #  #  #    #
 #      #    # #    # #    #
 #      #    # ###### #    #
 #      #    # #    # #    #
 ######  ####  #    # #####

"""


def load_population(
		rootdir : Path,
		modkeys : List[ModParam],
		modkeys_striponly : List[ModParam] = [],
		params_ref : Optional[ParamsDict] = None,
	) -> Population:

	# get all the individuals
	pathlst_indiv : List[Path] = [ 
		joinPath(rootdir,p)
		for p in os.listdir(rootdir)
		if (
			os.path.isdir(joinPath(rootdir,p))
			and p.startswith('h')
		)
	]

	# from each dir, arbitrarily pick the first params json file
	pathlst_params : List[Path] = [
		next(glob.iglob(joinPath(p,'**/params.json')))
		for p in pathlst_indiv
	]

	# load all the params files
	lst_paramdict : List[ParamsDict] = [
		load_params(p)
		for p in pathlst_params
	]

	# extract 
	lst_params_stripped : List[ParamsDict] = list()
	pop : Population = list()
	for p in lst_paramdict:
		p_strip,mod = extract_mods_from_params(
			params = p,
			modkeys = modkeys,
			modkeys_striponly = modkeys_striponly,
			default_val = float('nan'),
		)

		lst_params_stripped.append(p_strip)
		pop.append(mod)

	# TODO: check that params jsons match. this also means that strip-only keys dont mean anything at the moment
	if params_ref is not None:
		raise NotImplementedError('checking that params match is not yet implemented')
	# for p in lst_params_stripped:
	# 	for k,v in p.items():
	# 		if not isnan(v):
				
	return pop





"""
########  ##     ## ##    ##
##     ## ##     ## ###   ##
##     ## ##     ## ####  ##
########  ##     ## ## ## ##
##   ##   ##     ## ##  ####
##    ##  ##     ## ##   ###
##     ##  #######  ##    ##
"""


# UGLY: this function should just call the continuation function after initialization
def run_genetic_algorithm(
		# for setup
		rootdir : Path = "data/geno_sweep/",
		dists : ModParamsDists = DEFAULT_DISTS,
		first_gen_size : int = 50,
		gen_count : int = 20,
		factor_cull : float = 0.5,
		factor_repro : float = 2.0,
		# passed to `run_generation`
		path_params_base : Path = "input/chemo_v15.json",
		mut_sigma : float = 0.05,
		mutprob : float = 0.05,
		eval_runs : List[ModParamsDict] = DEFAULT_EVALRUNS,
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
		func_extract : ExtractorFunc = extract_food_dist_inv,
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
		verbose : bool = False,
	) -> None:
	"""runs a genetic optimization
	
	#### overview:
	- generates initial population using `first_gen_size` and `dists`
	- for each generation (using `run_generation`):
	  - runs simulations using C++ code `./sim.exe`, runs the sims given in `eval_runs`
	  - evaluates according to parameter `func_extract` and `calc_mean`
	  - culls population according to `factor_cull`
	  - creates new generation according to `factor_repro`, `mut_sigma`, `mut_prob` and `gene_combine`
	

	
	### Parameters:
	 - `rootdir : Path`   
	   stores sims and parameters in this directory (organized by generation)
	   (defaults to `"data/geno_sweep/"`)
	 - `dists : ModParamsDists`   
	   initial distribution of the parameters
	   (defaults to `DEFAULT_DISTS`)
	 - `first_gen_size : int`   
	   size of the first generated generations
	   (defaults to `500`)
	 - `gen_count : int`   
	   maximum number of generations to run for (terminating earlier is fine)
	   (defaults to `20`)
	 - `factor_cull : float`   
	   proportion of each population to preserve at each generation
	   (defaults to `0.5`)
	 - `factor_repro : float`   
	   size to multiply the culled population by to get the size of the next generation.
	   (defaults to `2.0`)
	 - `path_params_base : Path`   
	   [IMPORTANT] base set of parameters. Anything not overridden according to `dists` or `eval_runs` will match whats in this file
	   (defaults to `"input/chemo_v15.json"`)
	 - `mut_sigma : float`   
	   mutation follows a gaussian with sigma determined by multiplying `mut_sigma` by the range of parameters in the population (or maybe in `dists`? need to check)
	   (defaults to `0.05`)
	 - `mutprob : float`   
	   probability that a given parameter will be modified
	   (defaults to `0.05`)
	 - `eval_runs : List[ModParamsDict]`   
	   a list of parameters sets used to evaluate each worm genotype (i.e. try different angles)
	   (defaults to `DEFAULT_EVALRUNS`)
	 - `calc_mean : Callable[[List[float]], float]`   
	   for computing the fitness of the individual from multiple sims of the genotype. this is messy.
	   (defaults to `lambdax:min(x)`)
	 - `func_extract : ExtractorFunc`   
	   function to extract fitness from the output directory and parameters. see `_extract_TEMPLATE` in `pyutil.extract_run_data`
	   (defaults to `extract_food_dist_inv`)
	 - `gene_combine : GenoCombineFunc`   
	   function to combine two individuals into a single individual
	   (defaults to `combine_geno_select`)
	 - `gene_combine_kwargs : Dict[str,Any]`   
	   kwargs to pass to `gene_combine`
	   (defaults to `dict()`)
	 - `verbose : bool`   
	   verbose output printing (hashes and fitnesses for every individual)
	   (defaults to `False`)
	"""	

	params_base : ParamsDict = load_params(path_params_base)
	params_base["simulation"]["src-params"] = path_params_base

	mkdir(rootdir)
	with open(joinPath(rootdir, '.runinfo'), 'a') as info_fout:
		print('# info for run', file = info_fout)
		print(locals(), file = info_fout)
		print('\n\n', file = info_fout)

	# compute population sizes
	# TODO: `pop_sizes` should be an input parameter
	pop_sizes : List[Tuple[int, int]] = compute_gen_sizes(
		first_gen_size = first_gen_size,
		gen_count = gen_count,
		factor_cull = factor_cull,
		factor_repro = factor_repro,
	)
	prntmsg(f'computed population sizes for generations: \n\t{pop_sizes}')

	# compute ranges for mutation scaling
	ranges : ModParamsRanges = distributions_to_ranges(dists)

	# generate initial population
	pop : PopulationFitness = generate_geno(
		dists = dists,
		n_genos = pop_sizes[0][1],
	)

	prntmsg(f'generated initial population with {len(pop)} individuals')

	prntmsg(f'running generations')
	# run each generation
	for i,counts in enumerate(pop_sizes):
		count_cull,count_new = counts
		prntmsg(f'running generation {i} / {gen_count}, with population size {len(pop)} -> {count_cull} -> {count_new}', 1)
		
		generation_dir : Path = joinPath(rootdir, f"g{i}/")
		mkdir(generation_dir)

		pop = run_generation(
			pop = pop,
			rootdir = generation_dir,
			params_base = params_base,
			popsize_select = count_cull,
			popsize_new = count_new,
			ranges_override = ranges,
			mut_sigma = mut_sigma,
			mutprob = mutprob,
			eval_runs = eval_runs,
			calc_mean = calc_mean,
			func_extract = func_extract,
			gene_combine = gene_combine,
			gene_combine_kwargs = gene_combine_kwargs,
			n_gen = i,
			verbose = verbose,
		)

	# return final generation
	with open(joinPath(rootdir, '.runinfo'), 'a') as info_fout:
		print('## after run completion', file = info_fout)
		print(locals(), file = info_fout)
		print('\n\n', file = info_fout)
	
	# REVIEW: this return
	# return pop[0]



"""

  ####   ####  #    # #####
 #    # #    # ##   #   #
 #      #    # # #  #   #
 #      #    # #  # #   #
 #    # #    # #   ##   #
  ####   ####  #    #   #

"""

def continue_genetic_algorithm(
		# for setup
		rootdir : Path,
		dists : ModParamsDists = DEFAULT_DISTS,
		# first_gen_size : int = 500,
		gen_count : int = 20,
		factor_cull : float = 0.5,
		factor_repro : float = 2.0,
		# passed to `run_generation`
		path_params_base : Path = "input/chemo_v15.json",
		mut_sigma : float = 0.05,
		mutprob : float = 0.05,
		eval_runs : List[ModParamsDict] = DEFAULT_EVALRUNS,
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
		func_extract : ExtractorFunc = extract_food_dist_inv,
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
		verbose : bool = False,
	) -> None:

	params_base : ParamsDict = load_params(path_params_base)

	if not os.path.isdir(rootdir):
		FileNotFoundError(f'directory to continue run from does not exist: {rootdir}')

	with open(joinPath(rootdir, '.runinfo'), 'a') as info_fout:
		print('# info for run (continued)', file = info_fout)
		print(locals(), file = info_fout)
		print('\n\n', file = info_fout)

	# load starting population (pick largest generation number)
	generation_dirs : Dict[int, Path] = {
		int(p.strip('/gen_ ')) : p
		for p in os.listdir(rootdir)
		if (
			os.path.isdir(joinPath(rootdir,p)) 
			and p.startswith('g')
		)
	}

	last_gen : int = max(generation_dirs.keys())
	last_gen_path : Path = joinPath(rootdir, generation_dirs[last_gen])
	last_gen_size : int = len([
		p
		for p in os.listdir(last_gen_path)
		if (
			os.path.isdir(joinPath(last_gen_path,p))
			and p.startswith('h')
		)
	])

	pop : PopulationFitness = [
		(p,float('nan'))
		for p in load_population(
			rootdir = last_gen_path,
			modkeys = list(dists.keys()),
			# modkeys_striponly = list(*list(m.keys()) for m in eval_runs)
			# params_ref = 
		)
	]

	# compute population sizes
	pop_sizes : List[Tuple[int, int]] = compute_gen_sizes(
		first_gen_size = last_gen_size,
		gen_count = gen_count,
		factor_cull = factor_cull,
		factor_repro = factor_repro,
	)
	prntmsg(f'computed population sizes for new generations: \n\t{pop_sizes}')

	# compute ranges for mutation scaling
	ranges : ModParamsRanges = distributions_to_ranges(dists)

	prntmsg(f'generated initial population with {len(pop)} individuals')

	prntmsg(f'running generations')
	# run each generation
	for i,counts in enumerate(pop_sizes):
		n_gen : int = i + last_gen + 1
		count_cull,count_new = counts
		prntmsg(f'running generation {n_gen} / {gen_count+last_gen}, with population size {len(pop)} -> {count_cull} -> {count_new}', 1)
		
		generation_dir : Path = joinPath(rootdir, f"g{n_gen}/")
		mkdir(generation_dir)

		pop = run_generation(
			pop = pop,
			rootdir = generation_dir,
			params_base = params_base,
			popsize_select = count_cull,
			popsize_new = count_new,
			ranges_override = ranges,
			mut_sigma = mut_sigma,
			mutprob = mutprob,
			eval_runs = eval_runs,
			calc_mean = calc_mean,
			func_extract = func_extract,
			gene_combine = gene_combine,
			gene_combine_kwargs = gene_combine_kwargs,
			n_gen = n_gen,
			verbose = verbose,
		)

	# return final generation
	with open(joinPath(rootdir, '.runinfo'), 'a') as info_fout:
		print('## after run completion', file = info_fout)
		print(locals(), file = info_fout)
		print('\n\n', file = info_fout)
	
	# return pop