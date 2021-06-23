"""
# genetic algorithm notes
- look into simulated annealing
- make a "lean" version with:
	- no printing of position/activation @ every timestep, only print at end
	- dont print anything except the final position to stdout
"""

from posixpath import join
from typing import *
import subprocess
import copy
import os
from math import dist,isnan
import random
import json

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore
from pydbg import dbg # type: ignore

if TYPE_CHECKING:
	from mypy_extensions import Arg
else:
	Arg = lambda t,s : t

from pyutil.util import (
	ModParam, ModParamsDists, ModTypes, NormalDistTuple, Path,mkdir,joinPath,
	strList_to_dict,ParamsDict,ModParamsDict,ModParamsRanges,
	RangeTuple,norm_prob,
	VecXY,dump_state,
	find_conn_idx,find_conn_idx_regex,
	genCmd_singlerun,
	dict_hash,load_params,dict_to_filename,modprmdict_to_filename,
	keylist_access_nested_dict,
	read_body_data,CoordsRotArr,
	prntmsg,
)

from pyutil.geno_distr import DEFAULT_DISTS,DEFAULT_EVALRUNS


Process = Any
Population = List[ModParamsDict]
PopulationFitness = List[
	Tuple[ModParamsDict, float]
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



def jointo_nan_eval_runs(eval_runs : List[ModParamsDict]) -> ModParamsDict:
	"""
	create a `ModParamsDict` that maps every key from `eval_runs` to float('nan')
	
	useful for then merging into params
	"""
	joined_nans : ModParamsDict = dict()

	for x in eval_runs:
		for k in x.keys():
			joined_nans[k] = float('nan')

	return joined_nans

	



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

MultiExtractorFunc = Callable[
	[
		Arg(Path, 'datadir'),
		Arg(ParamsDict, 'params'),
		Arg(bool, 'ret_nan'),
	], 
	float, # return type
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

	result : ExtractorReturnType = func_extract(
		datadir = outpath,
		params = params_joined,
		ret_nan = bool(proc.returncode),
	)

	with open(joinPath(outpath, 'extracted.txt'), 'a') as fout_ext:
		print(f'# extracted using {func_extract.__name__}:', file = fout_ext)
		print(repr(result), file = fout_ext)
	
	return result


def _wrap_multi_extract(
		func_extract : ExtractorFunc,
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
	) -> MultiExtractorFunc:

	def _func_extract_MULTI(
			datadir : Path,
			params : ParamsDict,
			ret_nan : bool = False,
		) -> float:

		# TODO: ret_nan is not actually doing the correct thing here, although its probably unimportant. current implementation does not allow for just one of several processes failing

		lst_extracted : List[float] = list()

		for p in os.listdir(datadir):
			p_joined : Path = joinPath(datadir,p)
			if os.path.isdir(p_joined):
				lst_extracted.append(func_extract(
					datadir = p_joined,
					params = params,
					ret_nan = ret_nan,
				))

		return calc_mean(lst_extracted)

	# add metadata
	_func_extract_MULTI.__name__ = func_extract.__name__
	_func_extract_MULTI.__doc__ = f"""
		{_wrap_multi_extract.__doc__} 
		#### docstring of wrapped function:
		```markdown
		{func_extract.__doc__}
		```
	"""
	return _func_extract_MULTI


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
		bodydata : CoordsRotArr = read_body_data(joinPath(datadir,'body.dat'))[-1,0]
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
		bodydata : CoordsRotArr = read_body_data(joinPath(datadir,'body.dat'))[-1,0]
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
		[ key for key,_ in pop ], 
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

 ###### #    #   ##   #
 #      #    #  #  #  #
 #####  #    # #    # #
 #      #    # ###### #
 #       #  #  #    # #
 ######   ##   #    # ######

"""

def eval_pop_fitness(
		pop : Population,
		rootdir : Path,
		params_base : ParamsDict,
		func_extract : ExtractorFunc,
		eval_runs : List[ModParamsDict] = [ dict() ],
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
	) -> PopulationFitness:

	# wrap the extractor func for multiple runs
	func_extract_multi : MultiExtractorFunc = _wrap_multi_extract(func_extract)

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
	prntmsg(f'initialized {sum(len(x[2]) for x in to_read)} processes for {len(to_read)} individuals with unknown fitnesses:\n\t{" ".join(lst_ids)}\n', 2)

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

  ####  ###### #      ######  ####  #####
 #      #      #      #      #    #   #
  ####  #####  #      #####  #        #
      # #      #      #      #        #
 #    # #      #      #      #    #   #
  ####  ###### ###### ######  ####    #

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
	dbg(lst_fit)
	fitness_thresh : float = lst_fit[new_popsize]

	# TODO: WHYYYYYY is this line failing???
	"""
	File "F:\projects\Izq_locomotion\pyutil\genetic_utils.py", line 967, in run_generation
    pop_trimmed : PopulationFitness = generation_selection(pop, popsize_select)
	File "F:\projects\Izq_locomotion\pyutil\genetic_utils.py", line 907, in generation_selection
		prntmsg(f'distribution after trim: {str_fitness_distr(sorted([fit for prm,fit in newpop], reverse=True))}', 2)
	File "F:\projects\Izq_locomotion\pyutil\genetic_utils.py", line 907, in <listcomp>
		prntmsg(f'distribution after trim: {str_fitness_distr(sorted([fit for prm,fit in newpop], reverse=True))}', 2)
	ValueError: too many values to unpack (expected 2)
	"""

	prntmsg(f'fitness distribution: {str_fitness_distr(lst_fit)}', 2)
	prntmsg(f'trimming with fitness threshold approx {fitness_thresh}', 2)

	newpop : PopulationFitness = sorted(
		pop, 
		reverse = True, 
		key = lambda x : x[1],
	)[new_popsize]
	

	# newpop : PopulationFitness = [
	# 	(prm,fit)
	# 	for prm,fit in pop
	# 	if (fit > fitness_thresh)
	# ]

	dbg

	lst_fit_afterTrim : List[float] = sorted([fit for prm,fit in newpop], reverse=True)
	prntmsg(f'distribution after trim: {str_fitness_distr(lst_fit_afterTrim)}', 2)

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
		mut_sigma : float,
		mutprob : float,
		func_extract : ExtractorFunc,
		eval_runs: List[ModParamsDict],
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
		n_gen : int = -1,
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
	
	ranges_pop_mated : ModParamsRanges = get_pop_ranges([
		xa
		for xa,xb in pop_trimmed
	])

	pop_mutated : Population = [
		mutate_state(
			params_mod = prm,
			ranges = ranges_pop_mated,
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




def run_genetic_algorithm(
		# for setup
		rootdir : Path = "data/geno_sweep/",
		dists : ModParamsDists = DEFAULT_DISTS,
		first_gen_size : int = 500,
		gen_count : int = 10,
		factor_cull : float = 0.5,
		factor_repro : float = 2.0,
		# passed to `run_generation`
		params_base : ParamsDict = load_params("input/chemo_v14.json"),
		mut_sigma : float = 0.2,
		mutprob : float = 0.05,
		eval_runs : List[ModParamsDict] = DEFAULT_EVALRUNS,
		calc_mean : Callable[[List[float]], float] = lambda x : min(x),
		func_extract : ExtractorFunc = extract_food_dist_inv,
		gene_combine : GenoCombineFunc = combine_geno_select,
		gene_combine_kwargs : Dict[str,Any] = dict(),
	) -> PopulationFitness:

	mkdir(rootdir)
	with open(joinPath(rootdir, '.runinfo'), 'a') as info_fout:
		print('# info for run', file = info_fout)
		print(locals(), file = info_fout)
		print('\n\n', file = info_fout)

	# compute population sizes
	pop_sizes : List[Tuple[int, int]] = compute_gen_sizes(
		first_gen_size = first_gen_size,
		gen_count = gen_count,
		factor_cull = factor_cull,
		factor_repro = factor_repro,
	)
	prntmsg(f'computed population sizes for generations: \n\t{pop_sizes}')

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
		
		generation_dir : Path = joinPath(rootdir, f"gen_{i}/")
		mkdir(generation_dir)

		pop = run_generation(
			pop = pop,
			rootdir = generation_dir,
			params_base = params_base,
			popsize_select = count_cull,
			popsize_new = count_new,
			# ranges = ranges,
			mut_sigma = mut_sigma,
			mutprob = mutprob,
			eval_runs = eval_runs,
			calc_mean = calc_mean,
			func_extract = func_extract,
			gene_combine = gene_combine,
			gene_combine_kwargs = gene_combine_kwargs,
			n_gen = i,
		)

	# return final generation
	with open(joinPath(rootdir, '.runinfo'), 'a') as info_fout:
		print('## after run completion', file = info_fout)
		print(locals(), file = info_fout)
		print('\n\n', file = info_fout)
	
	return pop