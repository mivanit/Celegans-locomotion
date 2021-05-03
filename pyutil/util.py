import os
from typing import *

Path = str


def strList_to_dict(
		in_data : Union[dict,str], 
		keys_list : List[str], 
		delim : str = ',',
		type_map : Dict[str,Callable] = dict(),
	) -> Dict[str,Any]:
	if isinstance(in_data ,dict):
		return in_data
	else:
		# split into list
		in_lst : List[str] = in_data.split(delim)

		# map to the keys
		out_dict : Dict[str,Any] = {
			k:v 
			for k,v in zip(keys_list, in_lst)
		}

		# map types
		for key,func in type_map:
			if key in out_dict:
				out_dict[key] = func(out_dict[key])
		
		return out_dict





def joinPath(*args):
	return os.path.join(*args).replace("\\", "/")

SCRIPTNAME_KEY = "__main__"
COMMAND_DANGERS = [';', 'rm', 'sudo']

def _make_cmd_arg(arg : str, val : Optional[Any]) -> str:
	if val is None:
		return ""
	else:
		return f"--{arg} {val}"
	


def _command_assembler(**kwargs) -> str:
	output : List[str] = [ kwargs[SCRIPTNAME_KEY] ]

	for key,val in kwargs.items():
		if key != SCRIPTNAME_KEY:
			if val is not None:
				output.append(_make_cmd_arg(key, val))

	str_output : str = " ".join(output)
	
	for d in COMMAND_DANGERS:
		if d in str_output:
			print(f'WARNING: command contains: "{d}"\nplease review generated command:\n\t{str_output}\n')
			bln_cont : str = input('continue? (y/n)')
			if bln_cont != 'y':
				exit(0)

	return str_output


def genCmd_singlerun(
		params : Optional[Path] = None,
		coll : Optional[Path] = None,
		output : Optional[Path] = None,
		angle : Optional[float] = None,
		duration : Optional[float] = None,
		foodPos : Union[str, Tuple[float,float], None] = None,
		rand : Optional [bool] = None,
		seed : Optional [int] = None,
	) -> str:
	"""gets a shell command string for launching singlerun
	
	`./singlerun.exe <FLAGS>`
	
	### Parameters:
	 - `params : Optional[Path]`   
	   params json file
	   (defaults to `None`)
	 - `coll : Optional[Path]`   
	   collision tsv file
	   (defaults to `None`)
	 - `output : Optional[Path]`   
	   output dir
	   (defaults to `None`)
	 - `angle : Optional[float]`   
	   starting angle
	   (defaults to `None`)
	 - `duration : Optional[float]`   
	   sim duration in seconds
	   (defaults to `None`)
	 - `foodPos : Union[str, Tuple[float,float], None]`   
	   food position (comma separated) (defaults to whatever is in params.json). set to "DISABLE" to set the scalar to zero
	   (defaults to `None`)
	 - `rand : Optional[bool]`   
	   random initialization seed based on time
	   (defaults to `None`)
	 - `seed : Optional[int]`   
	   set random initialization seed. takes priority over `rand`. seed is 0 by default
	   (defaults to `None`)
	
	### Returns:
	 - `str` 
	   shell command
	"""

	cmd : str = _command_assembler(**{
		SCRIPTNAME_KEY : "./singlerun.exe",
		**locals(),
	}) 

	return cmd
	# return cmd + f' > {output}log.txt'
