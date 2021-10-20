from typing import *
import os
import sys

import numpy as np
from nptyping import NDArray

import matplotlib.pyplot as plt

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




def positions_to_heatmap(
		positions : ParticlePositions,
		bounds_tup : Optional[Tuple[float,float]] = None,
		gridpoints : int = 100,
		plot : bool = True,
		collobjs : Optional[str] = '../../input/objs/maze.tsv',
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
		# collision object stuff for bounds
		lst_collision_objects : List[CollisionObject] = list()
		if (collobjs is not None) and os.path.isfile(collobjs):
			lst_collision_objects = read_collobjs_tsv(collobjs)
		else:
			print(f'  >> WARNING: could not find file, skipping: {collobjs}')

		bounds_objs : BoundingBox = get_bounds(lst_collision_objects)

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
		
		# plot diffusion
		ax.imshow(
			H,
			extent = (xedges[0], xedges[1], yedges[0], yedges[1]),
			interpolation = 'bilinear',
		)
		
		# plot objects
		_plot_collobjs(ax, lst_collision_objects)
		
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


def read_and_plot(
		filename : str,
		bounds_tup : Optional[Tuple[float,float]] = None,
		gridpoints : int = 50,
	):
	data = np.load(filename)
	print(f'{data=}, {type(data)=}')

	# if old format, extend to 3rd dimension (0th idx is timestep)
	if len(data.shape) == 2:
		data = np.array([data])

	# assume that there are more than 2 particles, lol
	# and use this to fix the shape
	# since `positions_to_heatmap()` assumes first idx is particle idx, second is x/y
	if data.shape[1] == 2:
		data[:] = data[:].T

	print(data.shape)
	print(data) 

	for x in data:
		positions_to_heatmap(
			positions = x,
			bounds_tup = bounds_tup,
			gridpoints = gridpoints,
		)
		plt.show()


if __name__ == '__main__':
	import fire
	fire.Fire({
		'runsim' : run_particlesim_wrapper,
		'plot' : read_and_plot,
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

