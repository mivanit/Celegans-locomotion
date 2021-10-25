from typing import *
import os
import sys
import json

import numpy as np
from nptyping import NDArray

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numba

sys.path.append('../..')
from pyutil.collision_object import read_collobjs_tsv,CollisionObject
from pyutil.plot.pos import (
	_plot_collobjs,_get_fig_bounds_box,get_bounds,_combine_bounds,BoundingBox
)

# import cppyy
# cppyy.include('Collide_standalone.h')

ParticlePositions = NDArray[(Any, 2), float]

DEFAULT_PHYSICAL_PARAMS : Dict[str, Any] = {}


def initialize_particles(pos : Tuple[float,float], n : int) -> ParticlePositions:
	"""initialize `n` particles at position `pos`
	
	this function exists so it can later be extended for more complicated initlizations
	"""
	nd_pos : NDArray[2, float] = np.array(pos, dtype=np.float64)
	return np.full((n,2), nd_pos)

@numba.njit(cache=True)
def dummy_func(positions : NDArray):
	return positions * 5.0

@numba.njit(cache=True)
def iterate_particles(
		positions : NDArray,
		mean_dist : float,
		p_static : float,
	) -> NDArray:
	"""iterate the particle positions through a random walk
	
	### Parameters:
	 - `positions : ParticlePositions`   
		The current particle positions.
	 - `mean_dist : float`   
		mean distance travelled by a particle before changing direction (poisson distribution)
	 -  `p_static : float`
		probability particle does not move
	### Returns:
	 - `ParticlePositions`   
		The new particle positions after a timestep
	"""
	n_particles : int = positions.shape[0]
	# each particle will move a random distance in a random direction
	dists : NDArray[n_particles, float] = np.random.poisson(lam = mean_dist, size = n_particles)
	
	static : NDArray[n_particles, float] = np.random.rand(n_particles)

	# static : NDArray[n_particles, float] = np.random.choice(
	# 	[0,1], 
	# 	n_particles, 
	# 	True, # replace = True
	# 	np.array([p_static, 1-p_static]), # p = ...
	# )

	# apply the probability that the particle does not move
	dists = dists * (static > p_static)

	thetas : NDArray[n_particles, float] = np.random.uniform(0, 2*np.pi, size = n_particles)
	# convert to cartesian coordinates
	pos_delta : NDArray = np.stack(
		(dists*np.cos(thetas), dists*np.sin(thetas)),
		axis = 1,
	)

	return positions + pos_delta


@numba.njit(cache=True)
def run_particlesim(
		positions : NDArray,
		n_particles : int,
		mean_dist : float,
		p_static : float, 
		n_iterations : NDArray,
		save_every : int,
	) -> NDArray:

	output : NDArray = np.empty(
		(
			n_iterations // save_every,
			n_particles,
			2,
		), # shape
		np.int32,
	)

	for i in range(n_iterations):
		# positions = iterate_particles(
		# 	positions = positions,
		# 	mean_dist = mean_dist,
		# 	p_static = p_static,
		# )
		positions = dummy_func(
			positions,
		)
		
		if i % save_every == 0:
			output[i // save_every] = positions
	
	return output



@numba.njit(cache=True)
def run_particlesim_inline(
		positions : NDArray,
		n_particles : int,
		mean_dist : float,
		p_static : float, 
		n_iterations : NDArray,
		save_every : int,
	) -> NDArray:

	output : NDArray = np.empty(
		(
			n_iterations // save_every,
			n_particles,
			2,
		), # shape
		np.int32,
	)

	for i in range(n_iterations):
		dists : NDArray[n_particles, float] = np.random.poisson(lam = mean_dist, size = n_particles)		
		static : NDArray[n_particles, float] = np.random.rand(n_particles)
		dists = dists * (static > p_static)
		thetas : NDArray[n_particles, float] = np.random.uniform(0, 2*np.pi, size = n_particles)
		pos_delta : NDArray = np.stack(
			(dists*np.cos(thetas), dists*np.sin(thetas)),
			axis = 1,
		)

		# positions = positions + pos_delta
		
		if i % save_every == 0:
			output[i // save_every] = positions
	
	return output



class DiffusionDataSet(object):
	def __init__(
			self,
			posdata : Dict[int,ParticlePositions],
			metadata : Dict[str, Any],
		):
		self.posdata : ParticlePositions = posdata
		self.metadata : Dict[str,Any] = metadata

	def __getitem__(self, key : str) -> Any:
		return self.metadata[key]

	@staticmethod
	def read_from_basename(
			file_base : str,
			delete_old_type : bool = True,
			augment_final_timestep : bool = True,
		) -> 'DiffusionDataSet':
		"""reads from a base file
		
		### Parameters:
		- `file_base : str`   
		base file name -- `.npy` will be appended for pos data, `.json` for metadata
		
		### Returns:
		- `Dict[int,NDArray]`
		dict mappting iteration number to data, where first axis is particle idx and second is x/y
		"""

		fname_pos : str = file_base + '.npy'
		fname_meta : str = file_base + '.json'

		# load position data, and process
		posdata_raw : NDArray = np.load(fname_pos)
		# print(f'{posdata=}, {type(posdata)=}')

		# if old format, extend to 3rd dimension (0th idx is timestep)
		if len(posdata_raw.shape) == 2:
			posdata_raw = np.array([posdata_raw])

		# assume that there are more than 2 particles, lol
		# and use this to fix the shape
		# since `positions_to_heatmap()` assumes second idx is particle idx, third is x/y
		if posdata_raw.shape[1] == 2:
			posdata_raw[:] = posdata_raw[:].T

		# load metadata
		with open(fname_meta, 'r') as f:
			metadata : Dict[str,Any] = json.load(f)

		# split up the position data by timestep
		if len(metadata['tsteps']) != posdata_raw.shape[0]:
			raise ValueError(f"expected {len(metadata['tsteps'])} timesteps (from json metadata), got {posdata_raw.shape[0]} timesteps")

		posdata : Dict[int,ParticlePositions] = {
			t : posdata_raw[idx]
			for idx,t in enumerate(metadata['tsteps'])
		}
		# augment: -1 maps to final timestep
		if augment_final_timestep:
			posdata[-1] = posdata[max(posdata.keys())]

		# process collision objects
		metadata['collision_data'] = [
			CollisionObject.deserialize_json_dict(x)
			for x in metadata['collision_data']
		]

		return DiffusionDataSet(posdata, metadata)



def positions_to_heatmap(
		positions : ParticlePositions,
		bounds_tup : Optional[Tuple[float,float]] = None,
		gridpoints : int = 100,
		plot : bool = True,
		coll_objs : Optional[List[CollisionObject]] = None,
		tsteps : Optional[int] = None,
	) -> NDArray:

	if bounds_tup is None:
		bounds_tup = (
			np.min(positions),
			np.max(positions),
		)

	H,xedges,yedges = np.histogram2d(
		positions[:,0],
		positions[:,1],
		bins = gridpoints,
		range = (bounds_tup, bounds_tup),
	)


	if plot:
		if coll_objs is None:
			coll_objs = list()

		# collision object stuff for bounds
		bounds_objs : BoundingBox = get_bounds(coll_objs)

		# get bounds and set up figure
		bounds : BoundingBox = _combine_bounds([
			{
				'bound_min_x' : bounds_tup[0], 'bound_max_x' : bounds_tup[1],
				'bound_min_y' : bounds_tup[0], 'bound_max_y' : bounds_tup[1],
			},
			bounds_objs,
		])
		fig, ax = plt.subplots(1,1, figsize = _get_fig_bounds_box(bounds))
		ax.axis('equal')

		if tsteps is not None:
			ax.set_title(f'{tsteps=}')
		
		# plot diffusion
		ax.imshow(
			H,
			extent = (xedges[0], xedges[-1], yedges[0], yedges[-1]),
			# interpolation = 'bilinear',
		)
		
		# plot objects
		_plot_collobjs(ax, coll_objs)
		
		fig.show()

	return H



def run_particlesim_wrapper(
		n_particles : int = 1000000,
		mean_dist : float = 1.0,
		p_static : float = 0.2, 
		initial_pos : Tuple[float,float] = (0,0),
		n_iterations : int = 1000,
		save_every : int = 50,
	):

	# initial positions
	positions : ParticlePositions = np.full(
		(n_particles, 2), 
		fill_value = initial_pos,
		dtype = np.float32,
	)

	data = run_particlesim_inline(
		positions,
		n_particles,
		mean_dist,
		p_static,
		n_iterations,
		save_every,
	)

	for x in data:
		positions_to_heatmap(x)
		plt.show()



def plot_distributions_axis(
		basename : str,
		n_bins : int = 50,
		axis : int = 0,
	) -> None:
	"""plots the distribution of positions for all timesteps along the given axis
	
	### Parameters:
	 - `basename : str`   
	 - `n_bins : int` 
	   (defaults to `50`)
	 - `axis : int`   
	   0 for x, 1 for y
	   (defaults to `0`)
	"""

	data : DiffusionDataSet = DiffusionDataSet.read_from_basename(basename)

	colors_list : list = plt.cm.viridis(np.linspace(0, 1, len(data.posdata)))
	colors_map : dict = {
		t : colors_list[idx]
		for idx,t in enumerate(sorted(data.metadata['tsteps']))
		if t > 0
	}

	for t,pos in data.posdata.items():
		if t < 0:
			continue
		hist,bins = np.histogram(pos[:,axis], bins = n_bins)
		# get the centers of the bins
		bins = np.array([
			(bins[idx] + bins[idx+1]) / 2.0
			for idx in range(len(bins)-1)
		])

		plt.plot(
			bins,
			hist,
			'.-',
			color = colors_map[t],
			label = f'{t=}',
		)
	plt.legend()
	plt.show()

def read_and_plot(
		basename : str,
		bounds_tup : Optional[Tuple[float,float]] = None,
		gridpoints : int = 50,
		tsteps : Union[List[int],str,int,None] = None,
	):
	
	data : DiffusionDataSet = DiffusionDataSet.read_from_basename(basename)

	print(data.posdata[-1].shape)
	print(data.posdata[-1])

	if tsteps is None:
		tsteps = list(data.posdata.keys())
	elif isinstance(tsteps, str):
		tsteps = [ int(x) for x in tsteps.split(',') ]
	elif isinstance(tsteps, int):
		tsteps = [ tsteps ]

	for t,x in data.posdata.items():
		if t not in tsteps:
			continue
		positions_to_heatmap(
			positions = x,
			coll_objs = data.metadata['collision_data'],
			bounds_tup = bounds_tup,
			gridpoints = gridpoints,
			tsteps = t,
		)
		plt.show()

def plot_raw(
		basename : str,
		tsteps : Union[List[int],str,int,None] = None,
	):
	
	data : DiffusionDataSet = DiffusionDataSet.read_from_basename(basename)

	if tsteps is None:
		tsteps = list(data.posdata.keys())
	elif isinstance(tsteps, str):
		tsteps = [ int(x) for x in tsteps.split(',') ]
	elif isinstance(tsteps, int):
		tsteps = [ tsteps ]

	for t,x in data.posdata.items():
		if t not in tsteps:
			continue
		plt.plot(x[:,0], x[:,1], '.')
		plt.show()


if __name__ == '__main__':
	import fire
	# Make Python Fire not use a pager when it prints a help text
	fire.core.Display = lambda lines, out: print(*lines, file=out)
	fire.Fire({
		'runsim' : run_particlesim_wrapper,
		'plot' : read_and_plot,
		'plot_ax' : plot_distributions_axis,
		'plot_raw' : plot_raw,
	})






























	
def something_copilot_made(positions, velocities, dt, d, D, gamma, kT,
							   friction):
	"""
	Iterate the positions of the particles using the Langevin equation.

	Parameters
	----------
	positions : ndarray, shape (n, d)
		The current particle positions in d dimensions.
	velocities : ndarray, shape (n, d)
		The current particle velocities in d dimensions.
	dt : float
		The time step to use for the update.
	d : float
		The diffusion coefficient
	D : float
		The chemical potential
	gamma : float
		The drag coefficient
	kT : float
		The temperature in energy units.
	friction : float
		A damping factor on the velocity.  This is the gamma in the Langevin
		equation.

	Returns
	-------
	new_positions : ndarray, shape (n, d)
		The new particle positions.
	new_velocities : ndarray, shape (n, d)
		The new particle velocities.
	"""
	n, d = positions.shape
	positions = np.array(positions)
	velocities = np.array(velocities)
	new_positions = np.empty_like(positions)
	new_velocities = np.empty_like(velocities)
	for i in range(n):
		new_positions[i] = positions[i] + velocities[i] * dt
		new_velocities[i] = (
			velocities[i] + dt * friction * (
				-velocities[i] +
				random_force(d, kT) +
				d * grad_potential(positions[i], D) +
				gamma * random_force(d, kT)
			)
		)
	return new_positions, new_velocities

