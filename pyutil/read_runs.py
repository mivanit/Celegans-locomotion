from typing import *
import os

import yaml

if TYPE_CHECKING or (__name__ == 'pyutil.aggregate_runs'):
	from pyutil.util import *
	from pyutil.params import load_params
	from pyutil.collision_object import read_collobjs_tsv
else:
	from util import *
	from params import load_params
	from collision_object import read_collobjs_tsv


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

Enable_RunComponents_all : List[RunComponent] = [
	'params',
	'collobjs',
	'log',
	'fitness',
	'pos_all',
	'act_all',
]

Enable_RunComponents_short : List[RunComponent] = [
	'params',
	'collobjs',
	'log',
	'fitness',
	'pos_head',
	'act_head',
]

Enable_RunComponents_min : List[RunComponent] = [
	'params',
	'collobjs',
	'log',
	'fitness',
]

BodyData = Annotated[
	NDArray[(int, int), CoordsRotArr],
	ShapeAnnotation('timestep', 'segment'),
]

def read_body_data(filename : Path) -> BodyData:
	"""reads given tsv file into a numpy array
	
	array is a 2-D structured array of type `CoordsRotArr`
	with `'x', 'y', 'phi'` fields for each segment
	so essentially 3-D, where first index is timestep, second is segment, and third/field is x/y/phi
	
	### Parameters:
	- `filename : Path`   
	filename to read
	
	### Returns:
	- `BodyData` 
	"""
	# read in
	data_raw : NDArray = np.genfromtxt(filename, delimiter = ' ', dtype = None)

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
	) -> NDArray:
	
	# read the data
	data_df : pd.DataFrame = pd.read_csv(filename, sep = ' ')
	
	if nrn_filter is not None:
		# if filter is given, filter the columns
		cols_keep : List[str] = list(filter(
			nrn_filter,
			data_df.columns.values.tolist()
		))
		data_df = data_df[cols_keep]
	
	# convert to numpy array
	return data_df.to_records(index=False)

def load_old_txt_extracted(path: str) -> Dict[str, float]:
	# try to get the name of extracting function from the comment line
	with open(joinPath(path, 'extracted.txt'), 'r') as f:
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
	if not rootdir.is_dir():
		raise FileNotFoundError(f'{rootdir} is not a directory')
	
	if os.path.isfile(joinPath(rootdir, 'extracted.yml')):
		with open(joinPath(rootdir, 'extracted.yml'), 'r') as f:
			return yaml.safe_load(f)
	elif os.path.isfile(joinPath(rootdir, 'extracted.txt')):
		return load_old_txt_extracted(joinPath(rootdir, 'extracted.txt'))
	else:
		raise FileNotFoundError(f'neither extracted.yml nor extracted.txt found in {rootdir}')




read_runcomp_map : Dict[RunComponent, Callable[[Path], Any]] = {
	'params' : lambda p : load_params(joinPath(p, "params.json")),
	'collobjs' : lambda p : [ 
		x.serialize_lst() 
		for x in read_collobjs_tsv(joinPath(p, "coll_objs.tsv"))
	],
	'log' : lambda p : read_file(joinPath(p, "log.txt")),
	'fitness' : load_extracted,
	'pos_head' : lambda p : read_body_data(joinPath(p, "body.dat"))[:,0],
	'pos_all' : lambda p : read_body_data(joinPath(p, "body.dat")),
	'act_head' : ,
	'act_all' : ,
}




# def load_single_run(
# 		rootdir : Path,
# 		enable : Dict[RunComponent, bool] = {x:True for x in LST_RunComponents},
# 	) -> Dict[str, Any]:)