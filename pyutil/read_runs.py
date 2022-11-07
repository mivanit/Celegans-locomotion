from io import TextIOWrapper
from typing import *
import os
import sys
# from dataclasses import dataclass
import glob
import json
import pickle
import gzip

from pydbg import dbg # type: ignore

import numpy as np # type: ignore
from nptyping import NDArray # type: ignore
# from numpy.typing import NDArray

try:
  import msgpack # type: ignore
  import msgpack_numpy # type: ignore
  msgpack_numpy.patch()
except (ImportError,ModuleNotFoundError) as e:
  print(e)
  print('replacing msgpack with json')
  import json as msgpack
  

import yaml # type: ignore

__EXPECTED_PATH__ : str = 'pyutil.read_runs'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import *
from pyutil.params import load_params,ModParamsHashable
from pyutil.collision_object import read_collobjs_tsv


RunComponent = Literal[
	'params',
	'collobjs',
	'log',
	'fitness',
	'pos_head',
	'pos_all',
	'act_head',
	'act_all',
]

LST_RunComponents : Tuple[RunComponent, ...] = get_args(RunComponent)

ENABLE_RUNCOMPONENTS : Dict[str, List[RunComponent]] = {
	'all' : [
		'params',
		'collobjs',
		'log',
		'fitness',
		'pos_all',
		'act_all',
	],
	'short-act' : [
		'params',
		'collobjs',
		'log',
		'fitness',
		'pos_head',
		'act_head',
	],
	'short' : [
		'params',
		'collobjs',
		'log',
		'fitness',
		'pos_head',
	],
	'min' : [
		'params',
		'collobjs',
		'log',
		'fitness',
	],
}

# BodyData = NDArray[(Any, Any), CoordsRotArr]
# BodyData = Annotated[NDArray[(Any, Any), CoordsRotArr], ShapeAnnotation(('timestep', 'segment')) ]

def read_body_data(filename : Path):
	"""reads given tsv file into a numpy array
	
	array is a 2-D structured array of type `CoordsRotArr`
	with `'x', 'y', 'phi'` fields for each segment
	so essentially 3-D, where first index is timestep, second is segment, and third/field is x/y/phi
	
	### Parameters:
	- `filename : Path`   
	filename to read
	
	### Returns:
	- `NDArray[(Any, Any), CoordsRotArr]` 
	"""
	# read in
	try:
		data_raw : NDArray = np.genfromtxt(filename, delimiter = ' ', dtype = None)
	except ValueError as e:
		print(f'fail to read{filename}')
		print(e)
		data_raw = np.zeros((100,10))

	# trim first variable (time)
	data_raw = data_raw[:,1:]

	# compute dims
	n_tstep = data_raw.shape[0]
	n_seg = data_raw.shape[1] // 3

	# reshape to allow casting to structured array
	data_raw = np.reshape(data_raw, (n_tstep, n_seg, len(CoordsRotArr))) # type: ignore

	return recfunctions.unstructured_to_structured(
		data_raw,
		dtype = CoordsRotArr,
	)

	
READ_ACT_FILTERS : Dict[str, Callable[[str], bool]] = {
	'all' : lambda x: True,
	'head' : lambda x: (not ':' in x) or (x == 't'),
	'nrn' : lambda x: (not '.' in x) or (x == 't'),
	'body' : lambda x: (':' in x) or (x == 't'),
}

def read_act_data(
		filename : Path, 
		nrn_filter : Optional[Callable[[str], bool]] = None,
		replace_colnames : Optional[Tuple[str,str]] = None,
	) -> NDArray:
	
	# read the data
	data_df : pd.DataFrame = pd.read_csv(filename, sep = ' ')

	# fix the column names, because serialization of numpy structured arrays in messagepack is weird
	# colons are not allowed, so we replace them with dashes
	# -----
	# actually, messagepack sort of suck so we aren't going to do that anymore for consistency. 
	# the option still exists, just figure out how to set `replace_colnames = (':', '-')`
	# oh and also youd need to modify `READ_ACT_FILTERS`
	if replace_colnames is not None:
		data_df.columns = data_df.columns.str.replace(*replace_colnames)
	
	if nrn_filter is not None:
		# if filter is given, filter the columns
		cols_keep : List[str] = list(filter(
			nrn_filter,
			data_df.columns.values.tolist()
		))
		data_df = data_df[cols_keep]
	
	# convert to numpy array
	return data_df.to_records(index=False)


def load_old_txt_extracted(filename : str) -> Dict[str, float]:
	# try to get the name of extracting function from the comment line
	with open(filename, 'r') as f:
		output : Dict[str,float] = dict()
		
		# separate into comment blocks
		blocks : List[str] = f.read().split('#')
		blocks = [b for b in blocks if len(b) > 0]

		for b in blocks:
			# remove empty blocks
			if len(b.strip()) < 0:
				continue
			
			# split block `b` by colon
			b_parts : List[str] = b.split(':')

			if len(b_parts) != 2:
				continue

			# extract the name of the function
			func_name : str = b_parts[0].strip().split(' ')[-1].strip()

			# extract the output
			output[func_name] = float(b_parts[1].strip())

	return output


def load_extracted(rootdir : Path) -> Dict[str, float]:
	if not os.path.isdir(rootdir):
		raise FileNotFoundError(f'{rootdir} is not a directory')
	if os.path.isfile(joinPath(rootdir, 'extracted.yml')):
		with open(joinPath(rootdir, 'extracted.yml'), 'r') as f:
			return yaml.safe_load(f)
	elif os.path.isfile(joinPath(rootdir, 'extracted.txt')):
		return load_old_txt_extracted(joinPath(rootdir, 'extracted.txt'))
	else:
		raise FileNotFoundError(f'neither extracted.yml nor extracted.txt found in {rootdir}')




READ_RUNCOMP_MAP : Dict[RunComponent, Callable[[Path], Any]] = {
	'params' : lambda p : load_params(joinPath(p, "params.json")),
	'collobjs' : lambda p : [ 
		x.serialize_lst() 
		for x in read_collobjs_tsv(joinPath(p, "coll_objs.tsv"))
	],
	'log' : lambda p : read_file(joinPath(p, "log.txt")),
	'fitness' : load_extracted,
	'pos_head' : lambda p : read_body_data(joinPath(p, "body.dat"))[:,0],
	'pos_all' : lambda p : read_body_data(joinPath(p, "body.dat")),
	'act_head' : lambda p : read_act_data(
		joinPath(p, "act.dat"), 
		nrn_filter = READ_ACT_FILTERS['head'],
	),
	'act_all' : lambda p : read_act_data(joinPath(p, "act.dat")),
}

def validate_params(rootdir : Path, level : Literal["error", "warn", "quiet"] = "error") -> bool:
	try:
		params_json : Dict[str, Any] = load_params(joinPath(rootdir, "params.json"))
		params_json_alt : Dict[str, Any] = load_params(joinPath(rootdir, "in-params.json"))
	
		raise NotImplementedError("validation not yet done oops")

	except FileNotFoundError as e:
		if level == "error":
			raise e
		elif level == "warn":
			print(f'WARNING: missing one of two params files in {rootdir}: {e}\n')

		return False

		


def load_single_run(
		rootdir : Path,
		*,
		enable : Iterable[RunComponent],
		strict : bool = False,
		validate_mode : Literal["error", "warn", "quiet", "none"] = "none",
	) -> Dict[RunComponent, Any]:
	"""loads a single run from the given directory, attempting to get all data in `enabled`
	
	### Parameters:
	- `rootdir : Path`   
	path to the run directory
	- `enable : Dict[RunComponent, bool]` 
	dictionary of which components to load
	- `strict : bool`
	whether to raise an error if a component is not found
	(defaults to `False`)
	- `validate_mode : Literal["error", "warn", "quiet", "none"]`
	how to handle validation of params (not implemented)
	(defaults to "error")

	
	### Returns:
	- `Dict[RunComponent, Any]` 
	dictionary of run data
	"""
	# validate that the contetents of `params.json` match those of `in-params.json`
	if validate_mode != "none":
		validate_params(rootdir, validate_mode)

	# read in the run data
	run_data : Dict[RunComponent, Any] = dict()

	for run_component in LST_RunComponents:
		if run_component in enable:
			try:
				run_data[run_component] = READ_RUNCOMP_MAP[run_component](rootdir)
			except (FileNotFoundError,ValueError,IOError,KeyError) as e:
				if strict:
					raise e
				else:
					print(f'WARNING: {e}\n')
					run_data[run_component] = None
	
	return run_data


def read_evalruns_modparams(rootdir : Path) -> None:
	"""reads the parameters for the evaluation runs from the given directory
	
	modes:
	 - try to extract from the subirectory names
	 - try to extract the diffs between modparams in the runs
	 - try to extract from "eval_runs.json"
	"""
	raise NotImplementedError("`read_evalruns_modparams` not yet implemented")


def transform_dirname_evalruns(runs : List[str]) -> Dict[str, str]:
	"""translates old format seps '=/_' to new format '_/,'"""
	runs_map : Dict[str, str] = dict() 
	for x in runs:
		if '=' in x:
			runs_map[x] = (
				x
				.replace('_', ',')
				.replace('=', '_')
			)
		else:
			runs_map[x] = x

	return runs_map



def load_eval_run(
		rootdir : Path,
		*,
		enable : Iterable[RunComponent] = LST_RunComponents,
		strict : bool = False,
		validate_mode : Literal["error", "warn", "quiet", "none"] = "none",
	) -> Dict[Union[RunComponent, str], Any]:

	output : Dict[Union[RunComponent, str], Any] = dict()

	# load fitness data, if needed
	if 'fitness' in enable: 
		enable = {x for x in enable if x != 'fitness'}
		try:
			output['fitness'] = READ_RUNCOMP_MAP['fitness'](rootdir)				
		except FileNotFoundError as e:
			if strict:
				print("\n\n")
				raise e
				output['fitness'] = None
			else:
				print(f'\n\nWARNING: {e}')
				output['fitness'] = None

	# load the subdirectories
	lst_eval_dirs : List[Path] = [
		p 
		for p in os.listdir(rootdir) 
		if os.path.isdir(joinPath(rootdir, p))
	]
	
	dict_eval_runs : Dict[str,Any] = dict()

	for er_dir in lst_eval_dirs:
		dict_eval_runs[er_dir] = load_single_run(
			rootdir = joinPath(rootdir, er_dir),
			enable = enable,
			strict = strict,
			validate_mode = validate_mode,
		)

	output['eval_runs'] = dict_eval_runs

	# TODO: load eval runs params

	# TODO: store intersection of params, collobjs

	output['rootdir'] = unixPath(rootdir)

	return output


def load_recursive_allevals(
		rootdir : Path,
		enable : Iterable[RunComponent] = LST_RunComponents,
	) -> Dict[str, Any]:
	print(f'> searching in {rootdir}')
	alldirs : List[Path] = glob.glob(joinPath(rootdir,'**/h*/'), recursive=True)
	len_alldirs : int = len(alldirs)
	print(f'> found {len_alldirs} eval run directories\n')

	output : Dict[str, Any] = dict()
	for idx,subdir in enumerate(alldirs):
		subdir_unix : Path = unixPath(subdir)
		print(f'    > loading eval run  {idx+1} / {len_alldirs}\t{subdir_unix}' + ' '*5, end = '\r')
		output[subdir_unix] = load_eval_run(
			rootdir = subdir,
			strict = False,
			enable = enable,
		)
	
	print("\n")

	return output


class NumpyEncoder(json.JSONEncoder):
	""" Special json encoder for numpy types
	
	https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

	modified to save structured array as dict of arrays, where keys to dict are field names
	"""
	
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			if obj.dtype.names is None:
				return obj.tolist()
			else:
				return {
					k : obj[k].tolist() 
					for k in obj.dtype.names
				}
		elif isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		return json.JSONEncoder.default(self, obj)

"""some stats on how the various methods compare, size-wise

```bash
$ pwd; ls -l
/f/projects/CE-learn/Izq_locomotion/data/geno_sweep/old_sweep/chemo_v17_2
total 389756
-rw-r--r-- 1 mivanit 197609 145708056 Jul 25 16:56 data.json
-rw-r--r-- 1 mivanit 197609 100715871 Jul 25 17:04 data.msgpack
-rw-r--r-- 1 mivanit 197609 100839939 Jul 25 17:09 data.pkl
-rw-r--r-- 1 mivanit 197609    438166 Jul 25 17:20 data_min.json
-rw-r--r-- 1 mivanit 197609  12201112 Jul 25 17:22 data_short.json
-rw-r--r-- 1 mivanit 197609   1724664 Jul 25 18:59 data_short.json.gz
-rw-r--r-- 1 mivanit 197609   5038011 Jul 25 17:47 data_short.mpk
-rw-r--r-- 1 mivanit 197609   3428458 Jul 25 18:58 data_short.mpk.gz
-rw-r--r-- 1 mivanit 197609   5082089 Jul 25 19:12 data_short.pkl
-rw-r--r-- 1 mivanit 197609   3441790 Jul 25 19:12 data_short.pkl.gz
-rw-r--r-- 1 mivanit 197609   7952314 Jul 25 19:10 data_short.yaml
-rw-r--r-- 1 mivanit 197609   4275504 Jul 25 19:10 data_short.yaml.gz
-rw-r--r-- 1 mivanit 197609   6622529 Jul 25 17:20 data_short_noidnt.json
-rw-r--r-- 1 mivanit 197609   1611771 Jul 25 18:58 data_short_noidnt.json.gz
drwxr-xr-x 1 mivanit 197609         0 Jun 24 00:35 g0/
```

note that this is before the json files stored structured arrays properly. 
it appears that the new method for structured arrays takes a bit less space, which i suppose is good.
"""

def scrape_runinfo(rootdir : Path) -> Dict[str, str]:
	"""search for '.runinfo' files in `rootdir` using glob,
	place them into a dictorionary mapping path to data as string"""
	
	print(f'> looking for .runinfo files in {rootdir}')
	output : Dict[str, str] = dict()

	lst_runinfo_files : List[Path] = glob.glob(joinPath(rootdir, '**/.runinfo'), recursive = True)
	len_lst_runinfo_files : int = len(lst_runinfo_files)
	print(f'> found {len_lst_runinfo_files} files')
	
	for idx,runinfo_file in enumerate(lst_runinfo_files):
		runinfo_file_unix : str = unixPath(runinfo_file)
		print(f'\t> loading runinfo file {idx+1} / {len_lst_runinfo_files}\t{runinfo_file_unix}' + ' '*30)
		with open(runinfo_file_unix, 'r') as f:
			output[runinfo_file_unix] = f.read()

	print("\n")

	return output

def scrape_runinfo_interface(
		rootdir : Path, 
		fmt : str = "json", 
		filename : Optional[Path] = None,
		zip : bool = False) -> None:
	"""wraps `scrape_runinfo` to allow saving data to a file"""
	
	data : Dict[str,str] = scrape_runinfo(rootdir)

	# figure out the filename
	if filename is None:
		filename = joinPath(rootdir, f'runinfo_data.{fmt}')


	# file_open_func : Callable[[str, str], TextIOWrapper] = open
	file_open_func : Callable = open

	
	# if zipping, adjust things
	if zip:
		filename += '.gz'
		file_open_func = gzip.open

	print(f'> saving data to: {filename}')

	# save the data 
	with file_open_func(filename, SAVE_MODES[fmt]) as file:
		SAVE_FUNCS[SAVE_FORMATS[fmt]](data, file)


SAVE_FORMATS : Dict[str,str] = {
	"json" : "json",
	"messagepack" : "mpk",
	"msgpack" : "mpk",
	"mpk" : "mpk",
	"pickle" : "pkl",
	"pkl" : "pkl",
	"yaml" : "yaml",
	"yml" : "yaml",
}

SAVE_MODES : Dict[str,str] = {
	"json" : "wt",
	"mpk" : "wb",
	"pkl" : "wb",
	"yaml" : "wt",
}

# SAVE_FUNCS : Dict[str, Callable[[Any, str], None]] = {
SAVE_FUNCS : Dict[str, Callable] = {
	'json': lambda x,f,**kw: json.dump(x, f, cls = NumpyEncoder, **kw),
	'mpk': msgpack.dump,
	'pkl' : pickle.dump,
	'yaml' : yaml.dump,
}

def cli_wrapper_runloaders(
		func_load : Callable,
	) -> Callable:
	
	def newfunc(
			rootdir : Path,
			fmt : str = "json",
			enable : str = "all",
			filename : Optional[Path] = None,
			zip : bool = False,
			*args, **kwargs,
		) -> None:
		
		# load the data
		data : Dict[str,Any] = func_load(
			rootdir = rootdir, 
			enable = ENABLE_RUNCOMPONENTS[enable],
			**kwargs,
		)

		# figure out the filename
		if filename is None:
			filename = joinPath(rootdir, f'data_{enable}.{fmt}')

		# file_open_func : Callable[[str, str], TextIOWrapper] = open
		file_open_func : Callable = open
		
		# if zipping, adjust things
		if zip:
			filename += '.gz'
			file_open_func = gzip.open

		print(f'> saving data to: {filename}')

		# save the data 
		with file_open_func(filename, SAVE_MODES[fmt]) as file:
			SAVE_FUNCS[SAVE_FORMATS[fmt]](data, file)
	
	return newfunc
	
if __name__ == '__main__':
	import fire # type: ignore

	fire.Fire({
		'single' : cli_wrapper_runloaders(load_single_run),
		'eval' : cli_wrapper_runloaders(load_eval_run),
		'recursive' : cli_wrapper_runloaders(load_recursive_allevals),
		'runinfo' : scrape_runinfo_interface,
	})



















