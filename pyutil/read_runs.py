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