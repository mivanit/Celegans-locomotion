"""
plots the position of a worm and environment through time

contains plotters for showing head position of a single or multiple worms, the worm body at a point in time, or an animation showing the movement of the worm
"""

import os
import sys
from typing import *
import glob

from math import degrees
import json

import numpy as np # type: ignore
import numpy.lib.recfunctions as rfn # type: ignore
from nptyping import NDArray,StructuredType # type: ignore

import matplotlib # type: ignore 
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore
from matplotlib.patches import Patch,Circle,Rectangle,Wedge # type: ignore
from matplotlib.collections import PatchCollection # type: ignore

import pandas as pd # type: ignore
# from pydbg import dbg # type: ignore


__EXPECTED_PATH__ : str = 'pyutil.plot.pos'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import (
	Path,joinPath,unixPath,
	CoordsArr,CoordsRotArr,
	get_last_dir_name,pdbg,
)

from pyutil.read_runs import read_body_data

from pyutil.collision_object import (
	CollisionType,CollisionObject,
	read_collobjs_tsv,
	BoundingBox,AxBounds,BOUNDS_TEMPLATE,
	get_bounds,get_bbox_ranges,pad_BoundingBox,
	_bounds_tuples_to_bbox,_combine_bounds,
)


# types
# ==================================================
# TODO: make this actually reference matplotlib.Axes
Axes = Any

OptInt = Optional[int]

WORM_RADIUS = 80e-6


"""
########   #######  ##     ## ##    ## ########   ######
##     ## ##     ## ##     ## ###   ## ##     ## ##    ##
##     ## ##     ## ##     ## ####  ## ##     ## ##
########  ##     ## ##     ## ## ## ## ##     ##  ######
##     ## ##     ## ##     ## ##  #### ##     ##       ##
##     ## ##     ## ##     ## ##   ### ##     ## ##    ##
########   #######   #######  ##    ## ########   ######
"""

def arr_bounds(
		arr : NDArray, 
		pad_frac : float = 0.0,
	) -> AxBounds:
	"""return the bounds of `arr` padded by some fraction of the range
	
	[extended_summary]
	
	### Parameters:
	 - `arr : NDArray`   
	   input array
	 - `pad_frac : float`   
	   multiplied by range to determine padding
	   (defaults to `0.0`)
	
	### Returns:
	 - `AxBounds` 
	   padded bounds
	"""
	arr_min : float = np.amin(arr)
	arr_max : float = np.amax(arr)
	
	arr_range : float = arr_max - arr_min

	arr_min = arr_min - arr_range * pad_frac
	arr_max = arr_max + arr_range * pad_frac

	return (arr_min, arr_max)



def _get_fig_bounds(
		collobjs : List[CollisionObject],
		arrbd_x_in : Optional[AxBounds] = None, 
		arrbd_y_in : Optional[AxBounds] = None,
		figsize_scalar : float = 6.0,
	) -> NDArray[2, float]:

	collobjs_bounds : Dict[str,float] = get_bounds(collobjs)

	# set up the figure object
	if arrbd_x_in is None:
		# arrbd_x = arr_bounds(data['x'])
		arrbd_x : AxBounds = (
			collobjs_bounds['bound_min_x'],
			collobjs_bounds['bound_max_x'],
		)

	else:
		arrbd_x = arrbd_x_in

	if arrbd_y_in is None:
		# arrbd_y = arr_bounds(data['y'])
		arrbd_y : AxBounds = (
			collobjs_bounds['bound_min_y'], 
			collobjs_bounds['bound_max_y'],
		)

	else:
		arrbd_y = arrbd_y_in
	
	print('> positional bounds:\t', arrbd_x, arrbd_y)

	figsize : NDArray[2, float] = np.array([
		arrbd_x[1] - arrbd_x[0],
		arrbd_y[1] - arrbd_y[0],
	])
	
	# print(f'> figsize:\t{figsize}')

	return figsize * figsize_scalar / max(figsize)


def _get_fig_bounds_box(
		bounds : BoundingBox,
		figsize_scalar : float = 6.0,
	) -> NDArray[2, float]:

	figsize : NDArray[2, float] = np.array(list(get_bbox_ranges(bounds)))

	return figsize * figsize_scalar / max(figsize)


# 	data : NDArray[Any, CoordsRotArr],
# ) -> Tuple[NDArray[Any, CoordsArr], NDArray[Any, CoordsArr]]:
def body_data_split_DV(
		data : NDArray,
	) -> Tuple[NDArray, NDArray]:
	"""splits a body data file into arrays of dorsal and ventral points
	
	takes in a `CoordsRotArr` (produced by `read_body_data()`)
	and splits it into arrays of `CoordsArr` by the same method as `WormView.m`
	by Cohen et al [1]

	```matlab
	R = D/2.0*abs(sin(acos(((0:Nbar-1)-NSEG./2.0)./(NSEG/2.0 + 0.2))));`
	```
	
	### Parameters:
	 - `data : NDArray[Any, CoordsRotArr]`   
	   [description]
	
	### Returns:
	 - `Tuple[NDArray[Any, CoordsArr], NDArray[Any, CoordsArr]]` 
	   [description]

	### References
	 - [1] Boyle, J. H., Berri, S. & Cohen, N. Gait Modulation in C. elegans: An Integrated Neuromechanical Model. Front. Comput. Neurosci. 6, (2012).
	 	https://www.frontiersin.org/articles/10.3389/fncom.2012.00010/full

	"""
	n_tstep : int = data.shape[0]
	n_seg : int = data.shape[1]

	worm_thickness : NDArray[n_seg, float] = (
		WORM_RADIUS / 2.0 * abs(
			np.sin(np.arccos(
				((np.linspace(0,n_seg,n_seg)) - n_seg / 2.0) 
				/ ( n_seg / 2.0 + 0.2)
			))
		)
	)

	data_Dorsal : NDArray[(n_tstep, n_seg), CoordsArr] = np.full(
		shape = (n_tstep, n_seg),
		fill_value = np.nan,
		dtype = CoordsArr,
	)

	data_Ventral : NDArray[(n_tstep, n_seg), CoordsArr] = np.full(
		shape = (n_tstep, n_seg),
		fill_value = np.nan,
		dtype = CoordsArr,
	)

	# OPTIMIZE: this bit can be vectorized
	for t in range(n_tstep):
		dX : float = worm_thickness * np.cos(data[t]['phi'])
		dY : float = worm_thickness * np.sin(data[t]['phi'])
		data_Dorsal[t]['x'] = data[t]['x'] + dX
		data_Dorsal[t]['y'] = data[t]['y'] + dY   
		data_Ventral[t]['x'] = data[t]['x'] - dX   
		data_Ventral[t]['y'] = data[t]['y'] - dY 

	return (data_Dorsal, data_Ventral)


"""
        ########  ##        #######  ########
        ##     ## ##       ##     ##    ##
        ##     ## ##       ##     ##    ##
        ########  ##       ##     ##    ##
        ##        ##       ##     ##    ##
        ##        ##       ##     ##    ##
####### ##        ########  #######     ##
"""

def _plot_collision_boxes(ax : Axes, blocks : list, vecs : list):
	"""plots old-stype collision boxes

	### Parameters:
	 - `ax : Axes`   
	 - `blocks : list`   
	 - `vecs : list`   
	"""

	print(blocks)
	print(vecs)

	plot_boxes : List[Path] = []

	for bl in blocks:
		plot_boxes.append(Rectangle(
			xy = bl[0], 
			width = bl[1][0] - bl[0][0], 
			height = bl[1][1] - bl[0][1],
			fill = True,
		))

	pc : PatchCollection = PatchCollection(
		plot_boxes, 
		facecolor = 'red', 
		alpha = 0.5,
		edgecolor = 'red',
	)

	ax.add_collection(pc)


def _plot_collobjs(ax : Axes, collobjs : List[CollisionObject]):
	"""reads collision objects from a tsv file and plots them on `ax`
	
	### Parameters:
	 - `ax : Axes`   
	   matplotlib axes object
	 - `collobjs : List[CollisionObject]`   
	   list of collision objects (the kind where first entry is collider type)
	"""
	plot_objs : List[Patch] = []

	for obj in collobjs:
		if obj.coll_type == CollisionType.Box_Ax:
			plot_objs.append(Rectangle(
				xy = [obj['bound_min_x'], obj['bound_min_y']], 
				width = obj['bound_max_x'] - obj['bound_min_x'], 
				height = obj['bound_max_y'] - obj['bound_min_y'],
				fill = True,
			))
		elif obj.coll_type == CollisionType.Disc:
			plot_objs.append(Wedge(
				center = [ obj['centerpos_x'], obj['centerpos_y'] ],
				r = obj['radius_outer'],
				theta1 = degrees(obj['angle_min']),
				theta2 = degrees(obj['angle_max']),
				width = obj['radius_outer'] - obj['radius_inner'],

				fill = True,
			))

	pc : PatchCollection = PatchCollection(
		plot_objs, 
		facecolor = 'red', 
		alpha = 0.5,
		edgecolor = 'red',
	)

	ax.add_collection(pc)





def _plot_foodPos(
		ax : Axes, 
		params : Path, 
		fmt : str = 'x', 
		label : str = None, 
		maxdist_disc : bool = True,
	):
	with open(params, 'r') as fin:
		params_data : dict = json.load(fin)
		if "ChemoReceptors" in params_data:
			if "DISABLED" not in params_data["ChemoReceptors"]:
				foodpos_x : float = float(params_data["ChemoReceptors"]["foodPos"]["x"])
				foodpos_y : float = float(params_data["ChemoReceptors"]["foodPos"]["y"])
		
				ax.plot(foodpos_x, foodpos_y, fmt, label = label)

				if maxdist_disc:
					if "max_distance" in params_data["ChemoReceptors"]:
						ax.add_patch(Circle(
							(foodpos_x, foodpos_y), 
							radius = params_data["ChemoReceptors"]["max_distance"],
							alpha = 0.1,
							color = 'green',
						))
					else:
						KeyError('couldnt find "max_distance"')
			
				return (foodpos_x, foodpos_y)



"""
 ######  ######## ######## ##     ## ########
##    ## ##          ##    ##     ## ##     ##
##       ##          ##    ##     ## ##     ##
 ######  ######      ##    ##     ## ########
      ## ##          ##    ##     ## ##
##    ## ##          ##    ##     ## ##
 ######  ########    ##     #######  ##
"""


def _draw_setup(
		rootdir : Path = 'data/run/',
		bodydat : Path = 'body.dat',
		collobjs : Path = 'coll_objs.tsv',
		params : Optional[Path] = 'params.json',
		time_window : Tuple[OptInt,OptInt] = (None,None),
		figsize_scalar : Optional[float] = None,
		bounds : Optional[BoundingBox] = None,
		pad_frac : Optional[float] = None,
	) -> Tuple[
		plt.figure,
		Axes,
		NDArray[(Any,Any), CoordsRotArr],
		BoundingBox,
	]:
	"""sets up figure for drawing the worm
	
	- calculates bounding boxes (with padding)
	- sets up a scaled figure object
	- plots objects and food
	
	### Parameters:
	 - `rootdir : Path`   
		prepended to all other paths
	   (defaults to `'data/run/'`)
	 - `bodydat : Path`   
	   tsv of body data
	   (defaults to `'body.dat'`)
	 - `collobjs : Path`   
	   special tsv of collision objects
	   (defaults to `'coll_objs.tsv'`)
	 - `params : Optional[Path]`   
	   json of parameters, namely food position
	   (defaults to `'params.json'`)
	 - `time_window : Tuple[OptInt,OptInt]`   
	   will only preserve data between these two timesteps (also affects bounds)
	   (defaults to `(None,None)`)
	 - `figsize_scalar : float`   
	   sclar for figure size
	   (defaults to `6.0`)
	 - `bounds : Optional[BoundingBox]`   
	   manually specify bounds. if `None`, auto-generated
	   (defaults to `None`)
	 - `pad_frac : float`   
	   if auto generating bounds, pad the box by this fraction of the range on all sides
	   (defaults to `0.0`)
	
	### Returns:
	 - `Tuple[plt.figure, Axes, NDArray[(int,int), CoordsRotArr],BoundingBox]` 
	   returns figure (fig and ax objects), data, and bounding box of worm and objects
	"""

	# setting defaults
	# ==============================
	if figsize_scalar is None:
		figsize_scalar = 6.0
	if pad_frac is None:
		pad_frac = 0.0

	# getting the data
	# ==============================

	# prepend directory to paths
	bodydat = rootdir + bodydat
	collobjs = rootdir + collobjs
	params = rootdir + params if params is not None else None

	# read worm body
	data : NDArray[(int,int), CoordsRotArr] = read_body_data(bodydat)
	print(f'> raw data stats: shape = {data.shape}, \t dtype = {data.dtype}')
	# trim
	data = data[ time_window[0] : time_window[1] ]

	# read collision objects
	lst_collision_objects : List[CollisionObject] = list()
	if os.path.isfile(collobjs):
		lst_collision_objects = read_collobjs_tsv(collobjs)
	else:
		print(f'  >> WARNING: could not find file, skipping: {collobjs}')
		

	# get bounding boxes for contents
	# ==============================

	# get bounds
	bounds_objs : BoundingBox = get_bounds(lst_collision_objects)
	bounds_worm : BoundingBox = _combine_bounds([
		_bounds_tuples_to_bbox(arr_bounds(data['x'][:,0]), arr_bounds(data['y'][:,0])),
		_bounds_tuples_to_bbox(arr_bounds(data['x'][:,-1]), arr_bounds(data['x'][:,-1])),
	])

	# if no bounds given, replace with auto generated ones
	if bounds is None:
		bounds = _combine_bounds([bounds_objs, bounds_worm])

		# pad bounds
		bounds = pad_BoundingBox(bounds, pad_frac)
	
	print('test',bounds)

	# set up figure things
	# ==============================

	# get the figure size from the bounding box
	figsize : NDArray[2, float] = _get_fig_bounds_box(bounds, figsize_scalar)

	# create the figure itself
	print(f'> figsize:\t{figsize}')
	fig, ax = plt.subplots(1, 1, figsize = figsize)

	# fix the scaling
	ax.axis('equal')
	plt.title(rootdir)

	# plot preliminaries
	# ==============================

	# plot collision objects
	if collobjs is not None:
		_plot_collobjs(ax, lst_collision_objects)
	
	# plot food position
	if params is not None:
		#  and os.path.isfile(params):
		_plot_foodPos(ax, params)

	return (
		fig,ax,
		data,
		bounds,
	)





class Plotters(object):
	"""
	plots the position of a worm and environment through time

	contains plotters for showing head position of a single or multiple worms, the worm body at a point in time, or an animation showing the movement of the worm
	"""

	"""
	##     ## ########    ###    ########
	##     ## ##         ## ##   ##     ##
	##     ## ##        ##   ##  ##     ##
	######### ######   ##     ## ##     ##
	##     ## ##       ######### ##     ##
	##     ## ##       ##     ## ##     ##
	##     ## ######## ##     ## ########
	"""
	@staticmethod
	def pos(
			# args passed down to `_draw_setup()`
			rootdir : Path,
			bodydat : Path = 'body.dat',
			collobjs : Path = 'coll_objs.tsv',
			params : Optional[Path] = 'params.json',
			time_window : Tuple[OptInt,OptInt] = (None,None),
			figsize_scalar : Optional[float] = None,
			pad_frac : Optional[float] = None,
			# args specific to this plotter
			idx : int = 0,
			show : bool = True,
		):

		fig,ax,data,bounds = _draw_setup(
			rootdir = rootdir,
			bodydat = bodydat,
			collobjs = collobjs,
			params = params,
			time_window = time_window,
			figsize_scalar = figsize_scalar,
			pad_frac = figsize_scalar,
		)

		head_data : NDArray[data.shape[0], CoordsRotArr] = data[:,idx]

		print(head_data.shape, head_data.dtype)
		ax.plot(head_data['x'], head_data['y'])

		if show:
			plt.show()
		
	@staticmethod
	def pos_foodmulti(
			# search in this directory
			rootdir : Path,
			# args passed down to `_draw_setup()`
			bodydat : Path = 'body.dat',
			collobjs : Path = 'coll_objs.tsv',
			params : Optional[Path] = 'params.json',
			time_window : Tuple[OptInt,OptInt] = (None,None),
			figsize_scalar : Optional[float] = None,
			pad_frac : Optional[float] = None,
			# args specific to this plotter
			idx : int = 0,
			show : bool = True,
			food_excl : List[str] = [],
		):

		multi_dirs : List[str] = os.listdir(rootdir)
		multi_dirs = [ x for x in multi_dirs if os.path.isdir(rootdir + x) ]

		default_dir : str = joinPath(rootdir, multi_dirs[0]) + "/"
		print(f'> using as default: {default_dir}')

		fig,ax,data_default,bounds = _draw_setup(
			rootdir = default_dir,
			bodydat = bodydat,
			collobjs = collobjs,
			# params = params,
			time_window = time_window,
			figsize_scalar = figsize_scalar,
			pad_frac = figsize_scalar,
		)

		if isinstance(food_excl,str):
			food_excl = food_excl.split(',')

		for food_choice in multi_dirs:

			if food_choice not in food_excl:
			
				bodydat_choice : str = joinPath(rootdir, food_choice, bodydat)
				params_choice : str = joinPath(rootdir, food_choice, params)
							
				data : NDArray[(int,int), CoordsRotArr] = read_body_data(bodydat_choice)
				head_data : NDArray[data.shape[0], CoordsRotArr] = data[:,idx]

				print(bodydat_choice)
				print(head_data.shape, head_data.dtype)

				ax.plot(head_data['x'], head_data['y'], label = food_choice)
				tup_foodpos = _plot_foodPos(ax, params_choice, label = food_choice)
				print(tup_foodpos)

		plt.legend()

		if show:
			plt.show()

	@staticmethod
	def pos_multi(
			# search in this directory
			rootdir : Path,
			*args,
			# args passed down to `_draw_setup()`
			bodydat : Path = Path('body.dat'),
			collobjs : Path = Path('coll_objs.tsv'),
			params : Optional[Path] = Path('params.json'),
			time_window : Tuple[OptInt,OptInt] = (None,None),
			figsize_scalar : Optional[float] = None,
			pad_frac : Optional[float] = None,
			# args specific to this plotter
			idx : int = 0,
			show : bool = True,
			only_final : bool = False,
		):
		if not isinstance(rootdir, Path):
			rootdir = Path(rootdir)

		pdbg(rootdir)
		pdbg(bodydat)
		pdbg(rootdir / '**' / bodydat)
		lst_bodydat : List[Path] = glob.glob(rootdir / '**' / bodydat, recursive = True)
		lst_dirs : List[Path] = [ 
			unixPath(os.path.dirname(p)) + '/'
			for p in lst_bodydat
		]

		pdbg(lst_dirs)
		if not lst_dirs:
			raise FileNotFoundError('Could not find any matching files')
		default_dir : Path = lst_dirs[0]
		print(f'> using as default: {default_dir}')

		fig,ax,data_default,bounds = _draw_setup(
			rootdir = default_dir,
			bodydat = bodydat,
			collobjs = collobjs,
			# params = params,
			time_window = time_window,
			figsize_scalar = figsize_scalar,
			pad_frac = figsize_scalar,
		)

		for x_dir in lst_dirs:
			
			x_bodydat : str = joinPath(x_dir, bodydat)
			x_params : str = joinPath(x_dir, params)
						
			data : NDArray[(int,int), CoordsRotArr] = read_body_data(x_bodydat)
			
			head_data : NDArray[Any, CoordsRotArr] = data[-1,idx]
			if not only_final:
				head_data = data[:,idx]

			print(x_bodydat)
			print(head_data.shape, head_data.dtype)

			if only_final:
				ax.plot(head_data['x'], head_data['y'], 'o', label = x_dir)
			else:
				ax.plot(head_data['x'], head_data['y'], label = x_dir)
			# tup_foodpos = _plot_foodPos(ax, x_params, label = x_dir)
			# print(tup_foodpos)

		ax.set_title(rootdir / '**' / '')
		plt.legend()

		if show:
			plt.show()

	
	@staticmethod
	def pos_gener(
			*args,
			# search in this directory
			rootdir : Path,
			# args passed down to `_draw_setup()`
			bodydat : Path = 'body.dat',
			collobjs : Path = 'coll_objs.tsv',
			params : Optional[Path] = 'params.json',
			time_window : Tuple[OptInt,OptInt] = (None,None),
			figsize_scalar : Optional[float] = None,
			pad_frac : Optional[float] = None,
			# args specific to this plotter
			idx : int = 0,
			show : bool = True,
			max_gen : int = 5,
			gen_n_step : int = 1,
		):

		# setup
		lst_bodydat : List[Path] = glob.glob(joinPath(rootdir,bodydat), recursive = True)
		lst_dirs : List[Path] = [ 
			joinPath(os.path.dirname(p),'') 
			for p in lst_bodydat
		]

		default_dir : Path = lst_dirs[0]

		fig,ax,data_default,bounds = _draw_setup(
			rootdir = default_dir,
			bodydat = bodydat,
			collobjs = collobjs,
			# params = params,
			time_window = time_window,
			figsize_scalar = figsize_scalar,
			pad_frac = figsize_scalar,
		)

		for n_gen in range(0,max_gen+1,gen_n_step):
			# filter by generation
			lst_dirs_gen : List[Path] = [
				p
				for p in lst_dirs
				if get_last_dir_name(p,-3) == f'g{n_gen}'
			]

			print(f'  > for gen {n_gen} found {len(lst_dirs_gen)} dirs')

			head_data_x : List[float] = list()
			head_data_y : List[float] = list()

			for x_dir in lst_dirs_gen:
				
				x_bodydat : str = joinPath(x_dir, bodydat)
				x_params : str = joinPath(x_dir, params)
							
				data : NDArray[(int,int), CoordsRotArr] = read_body_data(x_bodydat)
				head_data_x.append(data[-1,idx]['x'])
				head_data_y.append(data[-1,idx]['y'])

			ax.plot(head_data_x, head_data_y, 'o', label = f'generation {n_gen}')

		plt.legend()

		if show:
			plt.show()

	"""
	   ###    ##    ## #### ##     ##
	  ## ##   ###   ##  ##  ###   ###
	 ##   ##  ####  ##  ##  #### ####
	##     ## ## ## ##  ##  ## ### ##
	######### ##  ####  ##  ##     ##
	##     ## ##   ###  ##  ##     ##
	##     ## ##    ## #### ##     ##
	"""

	@staticmethod
	def anim(
			rootdir : Path = 'data/run/',
			bodydat : Path = 'body.dat',
			collobjs : Path = 'coll_objs.tsv',
			params : Optional[Path] = 'params.json',
			output : Path = 'worm.mp4',
			time_window : Tuple[OptInt,OptInt] = (None,None),
			figsize_scalar : float = 6.0,
			fps : int = 30, bitrate : int = 1800,
		):
		"""
		https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
		credit to the above for info on how to use FuncAnimation
		"""
		output = rootdir + output
		# idk what this does tbh
		matplotlib.use("Agg")
		
		fig,ax,data,bounds = _draw_setup(
			rootdir = rootdir,
			bodydat = bodydat,
			collobjs = collobjs,
			params = params,
			time_window = time_window,
			figsize_scalar = figsize_scalar,
			pad_frac = figsize_scalar,
		)

		data_D, data_V = body_data_split_DV(data)
		
		# Set up formatting for the movie files
		writer : animation.FFMpegWriter = animation.writers['ffmpeg'](
			fps = fps, 
			metadata = dict(artist='Me'),
			bitrate = bitrate,
		)

		# this function gets called on each frame
		def anim_update(i, line_D, line_V):
			print(f'\t{i}\t/\t{data.shape[0]}', end = '\r')
			plt.title(f'frame   {i}')
			line_D.set_data(data_D[i]['x'], data_D[i]['y'])
			line_V.set_data(data_V[i]['x'], data_V[i]['y'])

			return line_D,line_V
		
		# set up the base worm
		line_D, = ax.plot([], [], 'r-')
		line_V, = ax.plot([], [], 'b-')

		print('> finished setup!')

		# make the animation
		line_ani = animation.FuncAnimation(
			fig, 
			anim_update,
			fargs = (line_D, line_V),
			frames = data.shape[0],
			interval = 50, 
			blit = True,
		)

		print(f'> animation created, saving to file `{output}`')

		# save it
		line_ani.save(output, writer = writer)

		print('\n\n> done saving!')

	"""
	######## ########     ###    ##     ## ########
	##       ##     ##   ## ##   ###   ### ##
	##       ##     ##  ##   ##  #### #### ##
	######   ########  ##     ## ## ### ## ######
	##       ##   ##   ######### ##     ## ##
	##       ##    ##  ##     ## ##     ## ##
	##       ##     ## ##     ## ##     ## ########
	"""

	@staticmethod
	def single_frame(
			rootdir : Path = 'data/run/',
			bodydat : Path = 'body.dat',
			collobjs : Path = 'coll_objs.tsv',
			params : Optional[Path] = 'params.json',
			arrbd_x = None, arrbd_y = None,
			i_frame : int = 0,
			figsize_scalar : float = 10.0,
			show : bool = True,
		) -> object:
		
		fig,ax,data,bounds = _draw_setup(
			rootdir = rootdir,
			bodydat = bodydat,
			collobjs = collobjs,
			params = params,
			time_window = (i_frame, i_frame+1),
			figsize_scalar = figsize_scalar,
			pad_frac = figsize_scalar,
		)

		data_D, data_V = body_data_split_DV(data)
		
		# set up the base worm
		line_D, = ax.plot(data_D[0]['x'], data_D[0]['y'], 'r-')
		line_V, = ax.plot(data_V[0]['x'], data_V[0]['y'], 'b-')

		if show:
			plt.show()

	@staticmethod
	def pos_multisegment(
			# args passed down to `_draw_setup()`
			rootdir: Path,
			bodydat: Path = 'body.dat',
			collobjs: Path = 'coll_objs.tsv',
			params: Optional[Path] = 'params.json',
			time_window: Tuple[OptInt, OptInt] = [None, None],
			figsize_scalar: Optional[float] = 10.0,
			pad_frac: Optional[float] = None,
			# args specific to this plotter
			idx: List = [0],
			show: bool = True,
	):

		fig,ax,data,bounds = _draw_setup(
			rootdir = rootdir,
			bodydat = bodydat,
			collobjs = collobjs,
			params = params,
			time_window = time_window,
			figsize_scalar = figsize_scalar,
			pad_frac = figsize_scalar,
		)

		if max(idx) >= max(data.shape):
			raise TypeError(f'The segment does not exist. The max frame is {data.shape}')

		if time_window[-1] is None and max(idx) != 0:
			time_window[-1] = max(idx)

		head_data : NDArray[data.shape[0], CoordsRotArr] = data[:max(idx), 0]


		# prepend directory to paths
		bodydat = os.path.join(rootdir, bodydat)
		data: NDArray[(int, int), CoordsRotArr] = read_body_data(bodydat)
		print(f'> raw data stats: shape = {data.shape}, \t dtype = {data.dtype}')

		for i_frame in idx:
			# trim
			data_local = data[i_frame: i_frame+1]
			data_D, data_V = body_data_split_DV(data_local)

			line_D, = ax.plot(data_D[0]['x'], data_D[0]['y'], 'r-')
			line_C, = ax.plot(data_local[0]['x'], data_local[0]['y'], 'k-')
			line_V, = ax.plot(data_V[0]['x'], data_V[0]['y'], 'b-')


		ax.plot(head_data['x'], head_data['y'])

		if show:
			plt.show()

if __name__ == '__main__':
	import fire # type: ignore
	fire.Fire(Plotters)


