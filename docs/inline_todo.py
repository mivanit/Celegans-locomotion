'''
inline_todo.py

https://github.com/knc-neural-calculus/knc-tools

helps you manage todos for a project

current features:
 - scrapes the files in a given directory for comments
 - outputs in a text file according to a chosen format

by Michael Ivanitskiy (miv@knc.ai)
'''


#* SETTINGS

import os
import sys
import glob
import re
import warnings
import json
from typing import *
from io import FileIO
from functools import cached_property
from collections import defaultdict

import yaml
import chevron # type: ignore
from omegaconf import OmegaConf,DictConfig,ListConfig

"""
 ######   #######  ##    ##  ######  ########  ######
##    ## ##     ## ###   ## ##    ##    ##    ##    ##
##       ##     ## ####  ## ##          ##    ##
##       ##     ## ## ## ##  ######     ##     ######
##       ##     ## ##  ####       ##    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##    ##
 ######   #######  ##    ##  ######     ##     ######
"""





# TODO: data from `file_in` is not being merged correctly
CONFIG_DEFAULT : DictConfig = OmegaConf.create({
	'config' : {
		'file_in' : 'itodo.yml',
		'file_out' : None,
	},
	'searchDir' : ['../modules', '../pyutil'],
	'file_todo' : 'todo-inline.md',
	'verbose' : False,
	# 'output' : {
	# 	'links'  	: True,
	# 	'tag-sort'  : True,
	# 	'file-sort' : True,
	# 	'dir-sort'  : True,
	# 	'format'	: 'md',
	# },
	'read' : {
		'tags' : {
			'list' : [
				"CRIT",
				"TODO",
				"FIXME",
				"FIX",
				"BUG",
				"DEBUG",
				"UGLY",
				"HACK",
				"NOTE",
				"IDEA",
				"REVIEW",
				"OPTIMIZE",
				"CONFIG",
				"!!!",
				"OLD"
			],
		},
		# 'comments' : {
		# 	'require' : False,
		# 	'list' : [
		# 		'//',
		# 		'/*'
		# 		'#',
		# 		'<!--',
		# 		'%',
		# 		'\\',
		# 	],
		# },
		# 'prefix' : {
		# 	'PREFIX' : '@',
		# 	'require' : False,
		# 	'always_accept' : False,
		# },
		'SOURCE_FILES' : [
			'c',
			'cpp',
			'h',
			'hpp',
			'py',
			'm',
			'tex',
			'sh',
			'java',
			'js',
		],
		'EXCLUDE' : [
			'inline_todo.py',
			'itodo.yml',
			'todo-inline.md',
		],
		'MAX_SEARCH_LEN' : 15,
		'context' : {
			'enabled' : True,
			'lines' : 5, # number of lines to show before and after the tag
		},
	},
	'write' : {
		'attr_sort_order' : ['tag', 'file', 'lineNum'],
		'item_format' : 'md_det',
	}
})



HEADER_YAML : Dict[str,Any] = {
	"title" : "todo-inline",
	"header-includes": """<style>
body {
  max-width: 50em;
}
</style>""",
}


# maps source file extensions to the corresponding file type
MAP_EXTENSION_TO_LANGUAGE : Dict[str, str] = {
	'c' : 'c',
	'h' : 'c',
	'cpp' : 'cpp',
	'hpp' : 'cpp',
	'cxx' : 'cpp',
	'hxx' : 'cpp',
	'py' : 'python',
	'm' : 'c',
	'tex' : 'latex',
	'tikz' : 'latex',
	'sh' : 'shell',
	'java' : 'java',
	'js' : 'javascript',
}


ItemPrintFormats = Literal['md', 'md_det']

ITM_FORMATS : Dict[ItemPrintFormats,str] = {
'md' : """ - [ ] {self.content} 
\t(line {self.lineNum})
""",
'md_det' : """ - [ ] {self.content} 
\t(line {self.lineNum})
\t
\t<details>
\t```{{.{lang} .numberLines startFrom="{startFrom}"}}
{self.context_processed}
\t```
\t</details>
""",
}


SortableAttrTodoItems = Literal[
	'tag',
	'file',
	'lineNum',
]

TEMPLATES : Dict[str,str] = { '3lvl' : r"""
{{#ul0}}
# {{h1}}
{{#ul1}}
## {{h2}}
{{#ul2}}
{{item}}
{{/ul2}}
{{/ul1}}
{{/ul0}}
""" }

def gen_template_from_attrlist(attrlist : List[str]) -> str:
	output : List[str] = list()

	depth : int = len(attrlist)

	# outermost list
	output.append(f'{{#ul0}}')

	# list openers, headers
	for idx in range(1, depth):
		output.append(f'{"#" * idx} {{h{idx}}}')
		output.append(f'{{#ul{idx}}}')

	# item
	output.append(r'{item}')

	# list closers
	for idx in range(depth-1, 0, -1):
		output.append(f'{{/ul{idx}}}')

	# outermost list closer
	output.append(f'{{/ul0}}')

	# join list
	# NOTE: moustance wants double curly braces
	return (
		'\n'
		.join(output)
		.replace('{', '{{')
		.replace('}', '}}')
	)	

class MustacheKey(object):
	# MAX_DEPTH : int = 1 + max(
	# 	TEMPLATE_BASE.count(r'{{#ul'),
	# 	len(TodoItem.Attr),
	# )
	
	# unordered_list = Literal[[f'ul{i}' for i in range(MAX_DEPTH)]]
	# header = Literal[[f'h{i}' for i in range(1, MAX_DEPTH + 1)]]
	
	unordered_list = Literal['ul0', 'ul1', 'ul2', 'ul3', 'ul4', 'ul5']
	header = Literal['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

	item = Literal['item']

	union_header_list = Union[header, unordered_list]

TEMPLATE_EXAMPLE_HASH : Dict[
		MustacheKey.unordered_list, 
		List[Dict[MustacheKey.union_header_list, Union[str,dict]]]
	] = { "ul0" : [
	{ 
		"h1": "TODO",
		"ul1": [
			{
				"h2": "foo.cpp",
				"ul2": [
					{ "item": "<class TodoItem>" },
					{ "item": "<class TodoItem>" }
				]
			},
			{
				"h2": "bar.py",
				"ul2": [
					{ "item": "<class TodoItem>" },
					{ "item": "<class TodoItem>" }
				]
			}
		]
	},
	{ 
		"h1": "HACK",
		"ul1": [
			{
				"h2": "foo.cpp",
				"ul2": [
					{ "item": "<class TodoItem>" },
					{ "item": "<class TodoItem>" }
				]
			},
			{
				"h2": "bar.py",
				"ul2": [
					{ "item": "<class TodoItem>" },
					{ "item": "<class TodoItem>" }
				]
			}
		]
	}
]}



def unixPath(path: str) -> str:
	return path.replace("\\", "/")

"""
 ######  ########  ######
##    ## ##       ##    ##
##       ##       ##
##       ######   ##   ####
##       ##       ##    ##
##    ## ##       ##    ##
 ######  ##        ######
"""


def assert_DictConfig(config: Any) -> DictConfig:
	if isinstance(config, DictConfig):
		return config
	else:
		raise TypeError(f"given config not a `DictConfig`, its type is {type(config)}")
		return OmegaConf.create({None : config})

def assert_Optional_DictConfig(config: Any) -> Optional[DictConfig]:
	if isinstance(config, DictConfig) or config is None:
		return config
	else:
		raise TypeError(f"given config not an `Optional[DictConfig]`, its type is {type(config)}")
		return OmegaConf.create({None : config})


def extract_frontmatter_yaml(filename : str, fm_delim : str = '---'):
	"""extracts YAML front matter from a markdown file
		
	REVIEW: so now, after spending several hours, I look in a separate cloned version of this project and find a commit that adds this function (to `itodo.py`, which im deleting in this commit), and remember that I wanted to store both configs and the written data in the same markdown file. this might not be the best idea, but maybe its worth looking into later. For now, the plan was to write metadata (count of items, files, lines, etc) in the yaml front matter. maybe store the config in there too though (but this would make it tough to write the yaml to an html file). either way, this function is still useful.

	### Parameters:
	 - `filename : str`   
	   markdown file
	 - `fm_delim : str`   
	   delimeter surrounding the front matter
	   (defaults to `'---'`)
	"""

	# read the appropriate lines from the file
	yaml_str : List[str] = []
	with open(filename, 'r') as fin:
		for line in fin:
			strpline = line.strip()
			if strpline == fm_delim:
				if len(yaml_str) > 0:
					# if its the end delimeter, stop reading
					break
			elif strpline:
				# if non empty string, add it
				yaml_str.append(strpline)
	
	# load the lines as  yaml
	return OmegaConf.create('\n'.join(yaml_str))


def add_default_excludes(config : DictConfig) -> None:
	"""adds default excludes to the config
	
	modifies `config['read']['EXCLUDE']`, by appending 
	```python
		config['file_todo'], # output file
		config['config']['file_in'], # input config file
		config['config']['file_out'], # input config file
		__file__, # this file
	```
	
	### Parameters:
	 - `config : DictConfig`
	   the config to modify
	
	### Returns:
	 - `None`
	
	"""
	config['read']['EXCLUDE'].extend([
		config['file_todo'], # output file
		config['config']['file_in'], # input config file
		config['config']['file_out'], # input config file
		__file__, # this file
	])

	# remove all non-strings from the exclude list
	config['read']['EXCLUDE'] = [
		x
		for x in config['read']['EXCLUDE']
		if isinstance(x, str)
	]

def load_file_config(filename : str) -> Optional[DictConfig]:
	"""loads config from a file, returns None and prints warning if load fails
	
	### Parameters:
	 - `filename : str`   
	
	### Returns:
	 - `Optional[DictConfig]` 
	   returns `None` if file is not found or another error occurs
	"""
	try:
		if os.path.isfile(filename):
			cfg_temp : Union[DictConfig, ListConfig] = OmegaConf.load(filename)
			cfg : DictConfig = assert_DictConfig(cfg_temp)

	except (FileNotFoundError,TypeError) as e:
		warnings.warn(f"loading config file failed:\t {filename} \n\tException caught, ignoring: {e}")
		return None

	return None


def process_configs(argv : List[str]) -> DictConfig:
	"""merge default, yaml, and CLI configs, return merged `DictConfig` object
		
	reads from:
	 - `CONFIG_DEFAULT`
	 - file specified by `CONFIG_DEFAULT['config']['file_in']`
	 - command line args
	 - file specified by command line arg `--config.file_in`
	
	Merges those configs into a single `OmegaConf.DictConfig` object, in the following order:
	 - `CONFIG_DEFAULT`
	 - file specified by `CONFIG_DEFAULT['config']['file_in']`
	 - file specified by command line arg `--config.file_in`
	 - command line args

	### Parameters:
	 - `argv : List[str]`   
	   command line arguments passed to `OmegaConf.from_cli()`
	
	### Returns:
	 - `DictConfig` 
	   merged config object
	"""	

	# default options
	cfg_default : DictConfig = assert_DictConfig(CONFIG_DEFAULT)

	# try to load config from default given file
	cfg_file_default : Optional[DictConfig] = assert_Optional_DictConfig(
		load_file_config(cfg_default['config']['file_in'])
	)
	
	# load command line arguments
	cfg_cmd : DictConfig = OmegaConf.from_cli(argv)

	# try to load config from cmd given file
	cfg_file_cmd : Optional[DictConfig] = None
	if ('config' in cfg_cmd) and ('file_in' in cfg_cmd['config']):
		cfg_file_cmd = assert_Optional_DictConfig(
			load_file_config(cfg_cmd['config']['file_in'])
		)

	# merge the configs (listed in increasing priority)
	tomerge : List[Optional[DictConfig]] = [
		cfg_default, cfg_file_default, cfg_file_cmd, cfg_cmd
	]

	tomerge_filtered : List[DictConfig] = [
		x
		for x in tomerge
		if x is not None
	]

	cfg = assert_DictConfig(OmegaConf.merge(*tomerge_filtered))

	# dont read this source file, or config files
	add_default_excludes(cfg)

	if cfg['config']['file_out'] is not None:
		# save the current (merged) config to the specified file
		OmegaConf.save(cfg, cfg['config']['file_out'])
		print('> saved config to\t%s' % cfg['config']['file_out'])

	return cfg


"""
 ######  ##        ######
##    ## ##       ##    ##
##       ##       ##
##       ##        ######
##       ##             ##
##    ## ##       ##    ##
 ######  ########  ######
"""



class TodoItem(object):

	Attr = Literal['tag', 'file', 'lineNum', 'line', 'context', 'content']

	def __init__(
			self,
			tag : str,
			file : str,
			lineNum : int,
			line : str,
			context : Optional[str] = None,
		) -> None:
		"""a class to hold a todo item
		
		### Parameters:
		 - `tag : str`   
		   tag of the todo item
		 - `file : str`   
		   file containing todo item
		 - `lineNum : int`   
		   line number of the todo item
		 - `line : str`   
		   content of the todo item's line
		 - `context : Optional[str]`
		   lines arond the todo item
		"""

		self.tag : str = tag
		self.file : str = file
		self.lineNum : int = lineNum
		self.line : str = line
		self.context : str = line if context is None else context

	def __str__(self) -> str:
		return f'[ {self.tag}\t:\t{self.file}\t:\t{self.lineNum} ]\t{self.content}'

	
	@cached_property
	def content(self) -> str:
		"""extract the content, trimming excess stuff

		reads until `self.tag` and then discards stuff

		note that this will probably break if more than one primary tag is present in a line
		this function does very little, but im writing it in anticipation of possible doing fancier things here such as extracting other tags (time created, assigned person, priority, etc)

		### Built-in Constants:
		 - `lstrip_chars : str`
		   characters to strip from the beginning of the `self.line` string, after we discard everything up to the tag
		   cant pass this a function because `content` is a cached property


		### Returns:
		- `str` 
		extracted content
		"""

		lstrip_chars : str = ':'

		idx = self.line.find(self.tag)
		output : str = self.line[idx:].lstrip(lstrip_chars).strip()

		return output

	@cached_property
	def context_processed(
			self, 
			tabs_spaces : int = 4) -> str:
		"""removes shared leading whitespace, adds a single leading tab"""
		# split by line, replace tabs with spaces
		context_lines : List[str] = [
			line.replace('\t', ' ' * tabs_spaces)
			for line in self.context.split('\n')
		]

		# calculate shared leading whitespace
		shared_whitespace_chars : int = min(
			len(s) - len(s.lstrip(' '))
			for s in context_lines
		)

		# remove shared whitespace
		context_lines = [
			line[shared_whitespace_chars:]
			for line in context_lines
		]

		# add a single tab to each line, for alignment inside code blocks
		context_lines = [
			'\t' + line
			for line in context_lines
		]

		# join with newline
		return '\n'.join(context_lines)		


	def to_str(self, fmt : ItemPrintFormats = 'md') -> str:
		if fmt == 'md':
			return ITM_FORMATS[fmt].format(self = self)
		elif fmt == 'md_det':
			return ITM_FORMATS[fmt].format(
				self = self,
				lang = MAP_EXTENSION_TO_LANGUAGE[
					self.file.split('.')[-1]
				],
				startFrom = self.lineNum,
			)




"""
########  ########    ###    ########
##     ## ##         ## ##   ##     ##
##     ## ##        ##   ##  ##     ##
########  ######   ##     ## ##     ##
##   ##   ##       ######### ##     ##
##    ##  ##       ##     ## ##     ##
##     ## ######## ##     ## ########
"""


def get_valid_files(
		searchdir : Union[str,List[str]], 
		file_types : List[str],
		exclude : Set[str] = set(),
	) -> List[str]:
	"""from `searchdir` return all files with types found in `file_types`
	
	### Parameters:
	 - `searchdir : Union[str,List[str]]`   
	   list of root directories to perform recursive search in
	 - `file_types : List[str]`   
	   file extensions
	 - `exclude : Set[str]`
	   anything that starts with a string in this set will be excluded
	   defaults to `set()`
	
	### Returns:
	 - `List[str]` 
	   list of valid filenames
	"""

	if isinstance(searchdir, str):
		searchdir = list([ searchdir ])

	files_search : List[str] = []
	
	for root in searchdir:
		# search individually for every file type
		for ft in file_types:
			glob_str_root = os.path.join(root, f'/*.{ft}')
			files_search.extend( glob.glob( glob_str_root, recursive = True ) )
			glob_str = os.path.join(root, f'**/*.{ft}')
			files_search.extend( glob.glob( glob_str, recursive = True ) )

	# make filepaths unix style
	files_search = [
		unixPath(x)
		for x in files_search
	]

	# filter out the files we need to ignore
	# TODO: do this using globs
	for ex in exclude:
		for root in searchdir:
			sw : str = os.path.join(root, ex)
			files_search = [
				unixPath(x)
				for x in files_search
				if not x.startswith(sw)
			]

	return files_search


def scrape_context(
		lineNum : int, 
		max_context_lines : int, 
		lst_lines : List[str],
		lst_lines_stripped : List[str],
	) -> str:

	# figure out when to read until
	# either until we reach the max context window,
	# or until the next empty line

	# find the index of the next empty line
	next_empty_line_idx : int = lineNum + max_context_lines
	try:
		next_empty_line_idx = lst_lines_stripped.index('', lineNum + 1, lineNum + max_context_lines)
	except ValueError:
		pass

	read_until : int = min(
		lineNum + max_context_lines,
		len(lst_lines) - 1,
		next_empty_line_idx,
	)							

	return '\n'.join(
		[ 
			x.strip('\n')
			for x in lst_lines[
				lineNum
				:lineNum + max_context_lines
			]
			if x.strip('\n') != ''
		]
	)



def scrape_items(
		filename : str,
		cfg_read : DictConfig,
	) -> List[TodoItem]:
	"""get a list of todo items from a file according to the settings in `cfg_read`
	
	### Parameters:
	 - `filename : str`   
	 - `cfg_read : DictConfig`   
	
	### Returns:
	 - `List[TodoItem]` 
	"""	

	td_items : List[TodoItem] = list()

		
	max_context_lines : Optional[int] = cfg_read['context']['lines'] if cfg_read['context']['enabled'] else None

	with open(filename, 'r', encoding='utf-8') as f:
		lst_lines : List[str] = f.readlines()
		lst_lines_stripped : List[str] = [
			x.strip()
			for x in lst_lines
		]

		# on every line, check for each tag
		for lineNum,line in enumerate(lst_lines):
			for tag in cfg_read['tags']['list']:
				if tag in line[:cfg_read['MAX_SEARCH_LEN']]:
					
					if max_context_lines is not None:
						context = scrape_context(
							lineNum = lineNum,
							max_context_lines = max_context_lines,
							lst_lines = lst_lines,
							lst_lines_stripped = lst_lines_stripped,
						)
					else:
						context = None

					# the first tag from `tags` will be used as the identifying one
					td_items.append(TodoItem(
						tag = tag,
						file = filename,
						lineNum = lineNum + 1,
						line = line,
						context = context,
					))
					break

	return td_items

def search_files(
		files_search : List[str],
		cfg_read : DictConfig,
	) -> List[TodoItem]:
	"""### search_files
	
	search files in `files_search` for valid todo comments
		
	### Parameters:
	 - `files_search : List[str]`   
	   list of filenames to search
	 - `cfg_read : DictConfig`
	   config 'read' section
	
	### Returns:
	 - `List[TodoItem]` 
	   list of inline comments scraped. not guaranteed to be in any particular order.
	"""	


	td_items : List[TodoItem] = list()

	# loop over every file
	for file in files_search:
		td_items.extend(scrape_items(file, cfg_read))
	
	return td_items


"""
########  ########   #######   ######
##     ## ##     ## ##     ## ##    ##
##     ## ##     ## ##     ## ##
########  ########  ##     ## ##
##        ##   ##   ##     ## ##
##        ##    ##  ##     ## ##    ##
##        ##     ##  #######   ######
"""


def split_byTag(td_items : List[TodoItem]) -> Dict[str, List[TodoItem]]:
	"""given a list of items, returns a dict of list where keys are tags
	
	REVIEW: `split_by_attribute()` does this and more, code should be removed at some point 

	"""
	items_byTag : Dict[str, List[TodoItem]] = dict()

	for x in td_items:
		if x.tag not in items_byTag:
			items_byTag[x.tag] = list()
		items_byTag[x.tag].append(x)
	
	return items_byTag


def get_sortkey_from_attr(attr : TodoItem.Attr) -> Callable[[TodoItem], Any]:
	"""return a function that will sort a list of items by a given attribute"""
	if attr == 'tag':
		# if `tag` is the sort key, sort by the order of tags in the default config
		return lambda x: CONFIG_DEFAULT['read']['tags']['list'].index(x.tag)
	elif attr == 'line':
		# if `line` is the sort key, sort by the order of lines in the file
		return lambda x: x.lineNum
	else:
		# otherwise, convert to string and sort
		return lambda x: str(x)

def split_by_attribute(
		td_items : List[TodoItem], 
		attr : TodoItem.Attr,
		filter_attr_values : Optional[List[str]] = None,
		sortKey : Optional[Callable[[TodoItem], Any]] = None,
	) -> Dict[str,List[TodoItem]]:
	"""given `td_items`, split the items up by their value of `attr`
		
	### Parameters:
	 - `td_items : List[TodoItem]`   
	   input list of todo items
	 - `attr : TodoItem.Attr`   
	   attribute of `TodoItem` by which to split,
	   should be one of `tag`, `file`, `lineNum`, `line`, or `content` -- but `tag` or `line` make the most sense
	 - `filter_attr_values : Optional[List[str]]`   
	   list of attribute values to incude, and the order in which they should appear in the output dict. If `None`, all attribute values will be incuded in arbitrary order. If a given attribute value has no corresponding elements, it will not be included in the output dict.
	   (defaults to `None`). 
	 - `sortKey : Optional[Callable[[TodoItem], Any]]`   
	   key for sorting the lists of todo items. If None, will try to figure out the best key based on the `attr` using `get_sortkey_from_attr()` (using order of tags in default config, for example).
	   (defaults to `None`)
	
	### Returns:
	 - `Dict[str,List[TodoItem]]` 
	   dictionary mapping possible values of `attr` to lists of todo items with that value
	"""	

	# try to figure out a smart sort key if none is given
	if sortKey is None:
		sortKey = get_sortkey_from_attr(attr)	

	# TODO: use defaultdict here
	items_byAttr : Dict[str, List[TodoItem]] = dict()
	
	# actually filter everything by the attribute value
	for itm in td_items:
		# get value of the attribute (this will work for properties too, not just actual existing attrs)
		a_val = itm.__getattribute__(attr)
		# handle default case
		if a_val not in items_byAttr:
			items_byAttr[a_val] = list()
		# append our item to the list
		items_byAttr[a_val].append(itm)

	
	# sort the lists
	for lst_items in items_byAttr.values():
		lst_items.sort(key = sortKey)
	
	# filter & sort the keys of the dict, if needed
	if filter_attr_values is None:
		return items_byAttr
	else:
		return {
			key : items_byAttr[key]
			for key in filter_attr_values if key in items_byAttr
		}
	



"""
##      ## ########  #### ######## ########
##  ##  ## ##     ##  ##     ##    ##
##  ##  ## ##     ##  ##     ##    ##
##  ##  ## ########   ##     ##    ######
##  ##  ## ##   ##    ##     ##    ##
##  ##  ## ##    ##   ##     ##    ##
 ###  ###  ##     ## ####    ##    ########
"""

"""Union[
	List[Dict[
		MustacheKey.union_header_list,
		Union[ str, list ]
	]],
	List[ Dict[ MustacheKey.item, str ] ],
]
"""

def _hdr_items_count(count : int) -> str:
	if count == 0:
		return 'no items'
	elif count == 1:
		return '1 item'
	else:
		return f'{count} items'

def format_attr_header(
		attr : SortableAttrTodoItems, 
		val : str,
		lst_items : List[TodoItem],
		lvl : int = 0,
	):
	"""format a header for a given attribute (links to files, etc)"""
	if attr == 'tag':
		return f'**{val}** -- {_hdr_items_count(len(lst_items))}'
	elif attr == 'file':
		return f'[`{val}`]({val}) -- {_hdr_items_count(len(lst_items))}'
	else:
		return val


def recursive_sortattr(
		td_items : List[TodoItem],
		attr_sort_order : List[SortableAttrTodoItems],
		lvl : int = 0,
		fmt : ItemPrintFormats = 'md',
	) -> Union[
		List[Dict[
			str, # MustacheKey.union_header_list,
			Union[ str, list ]
		]],
		List[Dict[ 
			str, # MustacheKey.item, 
			str 
		]],
	]:
	"""recursively sort `td_items` by the attributes in `attr_sort_order`
	
	split up each list into a dictionary whose keys are values for the attribute `attr_sort_order[lvl]` and whose values are lists of items with that value

	note that for the base case, items are still sorted by the last entry in `attr_sort_order` but not split up into headers (this is to allow sorting by line number)

	structure needs to be as follows:

	```json
	{json_example}
	```

	### Parameters:
	 - `td_items : List[TodoItem]`  
	   list of items
	 - `attr_sort_order : List[SortableAttrTodoItems]`   
	   a tuple of attributes to sort by
	 - `lvl : int`
	   the current level -- determines the keys and index into `attr_sort_order`
	 - `fmt : ItemPrintFormats`
	   format to print items in

	### Returns: (one of)
	 - `List[ Dict[ MustacheKey.item, TodoItem ] ]`
	   base case if `lvl == len(attr_sort_order)`
	 - `List[Dict[ MustacheKey.union_header_list, Union[str, list]]]`
	   recursive case
	"""

	current_attr : SortableAttrTodoItems = attr_sort_order[lvl]

	# base case: end of `attr_sort_order`
	if lvl >= len(attr_sort_order) - 1:
		return [
			{ 'item' : x.to_str(fmt = fmt) }
			for x in sorted(td_items, key = get_sortkey_from_attr(current_attr))
			# sort by whatever the final level attribute is
		]

	# sort by the corresponding attribute
	items_byAttr : Dict[str,List[TodoItem]] = split_by_attribute(
		td_items = td_items,
		attr = current_attr,
		# sortKey = lambda x : x.__getattribute__(current_attr),
	)

	# recurse
	return [
		{
			f'h{lvl+1}' : format_attr_header(
				attr = current_attr, 
				val = attrVal,
				lvl = lvl,
				lst_items = lst_items,
			),
			f'ul{lvl+1}' : recursive_sortattr(
				td_items = lst_items,
				attr_sort_order = attr_sort_order,
				lvl = lvl + 1,
				fmt = fmt,
			),
		}
		for attrVal,lst_items in items_byAttr.items()
	]

recursive_sortattr.__doc__ = str(recursive_sortattr.__doc__).format(json_example = json.dumps(TEMPLATE_EXAMPLE_HASH, indent = '\t'))


def write_items_ms_template(
		td_items : List[TodoItem], 
		item_format : ItemPrintFormats = 'md',
		attr_sort_order : List[SortableAttrTodoItems] = ['tag', 'file', 'lineNum'],
	) -> str:
	"""given a list of todo items, sort them, convert to strings, and plug into the template using chevron
	
	### Parameters:
	 - `td_items : List[TodoItem]`   
	   input list of todo items
	 - `item_format : ItemPrintFormats`
	   key to `ITM_FORMATS` dict to use for formatting each item
	   (defaults to `'md'`)
	 - `attr_sort_order : Optional[Tuple[str,...]]`
	   order in which to sort the tags and files before putting in the template
	
	### Returns:
	 - `str` 
	   formatted string
	"""

	# generate template of correct depth
	ms_template : str = gen_template_from_attrlist(attr_sort_order)

	# the first list needs to be added manually
	data : Dict[str,Any] = { 'ul0': 
		recursive_sortattr(
			td_items = td_items,
			attr_sort_order = attr_sort_order,
			fmt = item_format,
		)
	}

	# TODO: replace any chars that need to be escaped that arent inside code blocks

	# HACK: patches the chevron renderer to not replace html strings
	chevron.renderer._html_escape = lambda string: string

	rendered : str = chevron.render(
		ms_template,
		data,
	)

	return rendered



"""

 #      ######  ####    ##    ####  #   #
 #      #      #    #  #  #  #    #  # #
 #      #####  #      #    # #        #
 #      #      #  ### ###### #        #
 #      #      #    # #    # #    #   #
 ###### ######  ####  #    #  ####    #

"""

def write_markdown_byTag(
		td_items : List[TodoItem],
		fout,
		tags_print : Optional[List[str]] = None,
		rem_searchDir : str = '.',
	) -> None:
	"""write the given todo items to the given stream
	
	REVIEW: this function should eventually be replaced by `write_items_ms_template()`,
	but can be preserved as a fallback for when `chevron` is not installed

	prints in markdown format, with h1 headings for tags and h2 headings for files. 
	then, print ordered by line number
	this is meant to be a placeholder for before I implement a more flexible writing utility
	
	### Parameters:
	 - `td_items : List[TodoItem]`   
	   list of items
	 - `fout`   
	   output filestream. you can set this manually to `None` to have it print to console
	 - `tags_print : Optional[List[str]]`   
	   tags to print. probably dont use this option, just specify in config which tags to read in the first place
	   (defaults to `None`)
	"""	


	# define custom print function that prints to given stream
	printf = lambda *args, **kwargs : print(*args, **kwargs, file = fout)

	# print header
	printf('# inline TODO summary:')
	from datetime import datetime
	printf( 'Updated:\t' + str(datetime.now()))

	# first loop over tags
	items_byTag : Dict[str,List[TodoItem]] = split_by_attribute(td_items, 'tag', tags_print)
	for tag,tag_items in items_byTag.items():
		# print the header
		printf('\n# %s:' % tag)
		# and then each item
		items_byFile : Dict[str,List[TodoItem]] = split_by_attribute(
			td_items = tag_items,
			attr = 'file',
			sortKey = lambda x : x.lineNum,
		)

		for file,file_items in items_byFile.items():
			fname_to_print = file.replace(rem_searchDir + '/', '', 1)
			printf('\n## [`%s`](%s)' % (fname_to_print, file))
			for item in file_items:
				printf(item.to_str(fmt = 'md'))






"""
########  ##     ## ##    ##
##     ## ##     ## ###   ##
##     ## ##     ## ####  ##
########  ##     ## ## ## ##
##   ##   ##     ## ##  ####
##    ##  ##     ## ##   ###
##     ##  #######  ##    ##
"""

def main(argv):
	# load the config
	cfg : DictConfig = process_configs(argv)
	
	# get all valid files
	filenames : List[str] = get_valid_files(
		searchdir = cfg['searchDir'],
		file_types = cfg['read']['SOURCE_FILES'],
		exclude = set(cfg['read']['EXCLUDE']),
	)

	# get todo items from files
	todo_items : List[TodoItem] = search_files(
		files_search = filenames,
		cfg_read = cfg['read'],
	)

	# sort, put together
	# CRIT: there is some sort of bug with loading configs that I can't pin down
	data : str = write_items_ms_template(
		td_items = todo_items,
		item_format = cfg['write']['item_format'],
		attr_sort_order = cfg['write']['attr_sort_order'],
	)

	# put metadata (# items, # files, etc) in yaml header of output file
	metadata_dict : Dict[str,Any] = { 'metadata' : {
		'searched_files' : len(filenames),
		'files_with_todos' : len(set(x.file for x in todo_items)),
		'num_items' : len(todo_items),
		'num_unique_tags' : len(set(x.tag for x in todo_items)),
	}}

	# write to file
	# TODO: print to console if no output file specified
	with open(cfg['file_todo'], 'w', encoding = 'utf-8') as fout:
		print('---', file = fout)
		print(yaml.dump(HEADER_YAML), file = fout)
		print(yaml.dump(metadata_dict), file = fout)
		print('# suggested command for conversion to html', file = fout)
		print('cmd: "pandoc todo-inline.md -o todo-inline.html --from markdown+backtick_code_blocks+fenced_code_attributes --standalone --toc --toc-depth 1"', file = fout)
		print('---', file = fout)
		print(data, file = fout)



if __name__ == "__main__":
	main(sys.argv)

