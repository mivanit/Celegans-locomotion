#include <assert.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_map>

const double EPSILON = 0.00000000001;

enum CollisionType
{
	Box_Ax,
	Disc,
};

// collision object stuff
// REVIEW: this would be cleaner if I used polymorphism properly lol
struct CollisionObject
{
	CollisionType coll_type;

	// bounding box always used
	double bound_min_x;
	double bound_min_y;

	double bound_max_x;
	double bound_max_y;

	// only used if Box
	double fvec_x;
	double fvec_y;

	// only used if disc
	double centerpos_x;
	double centerpos_y;

	double force;
	double radius_inner;
	double radius_outer;

	double angle_min;
	double angle_max;

	std::unordered_map<std::string, double> as_umap()
	{
		std::unordered_map<std::string, double> ret;
		
		ret["_HACKY_coll_type"] = coll_type;
		ret["bound_min_x"] = bound_min_x;
		ret["bound_min_y"] = bound_min_y;
		ret["bound_max_x"] = bound_max_x;
		ret["bound_max_y"] = bound_max_y;

		if (coll_type == Box_Ax)
		{
			ret["__type__:Box_Ax"] = coll_type;

			ret["fvec_x"] = fvec_x;
			ret["fvec_y"] = fvec_y;
		}
		else if (coll_type == Disc)
		{
			ret["__type__:Disc"] = coll_type;

			ret["centerpos_x"] = centerpos_x;
			ret["centerpos_y"] = centerpos_y;

			ret["force"] = force;
			ret["radius_inner"] = radius_inner;
			ret["radius_outer"] = radius_outer;
			
			ret["angle_min"] = angle_min;
			ret["angle_max"] = angle_max;
		}

		return ret;
	}
};

// x-y vector stuff
struct VecXY
{
	double x;
	double y;

	VecXY(double in_x, double in_y)
	{
		this->x = in_x;
		this->y = in_y;
	}

	// VecXY(VecXY & in_vec)
	// {
	// 	this->x = in_vec.x;
	// 	this->y = in_vec.y;
	// }

	VecXY()
	{
		this->x = 0.0;
		this->y = 0.0;
	}

	bool is_nonzero()
	{
        return ((fabs(x) > EPSILON) || (fabs(y) > EPSILON));
		// NOTE: this printf statement is cursed. somehow `(fabs(y) > EPSILON) ? "true" : "false"` evaluated to "sin" before causing a segfault. no clue what was going on.
		// printf(
		// 	"is_nonzero:\t%f,%f,%f\t%s,%s\n", 
		// 	fabs(x), fabs(y), EPSILON,
		// 	(fabs(x) > EPSILON) ? "true" : "false", 
		// 	(fabs(y) > EPSILON) ? "true" : "false"
		// );
		// NOTE: for some reason, abs() doesnt work and casts things to ints
		// std::cout << std::fixed;
		// std::cout << std::setprecision(5) << "is_nonzero:\t" << fabs(x) << "," << fabs(y) << ","  << EPSILON << ","  << ((fabs(x) > EPSILON) ? "true" : "false") << ","  << ((fabs(y) > EPSILON) ? "true" : "false") << std::endl;
    }

	inline double mag()
	{
		return pow(
			( pow(x, 2.0) + pow(y, 2.0) ), 
			0.5
		);
	}

	void scale(double c)
	{
		x *= c;
		y *= c;
	}

	void normalize()
	{
		double mag = this->mag();
		if (mag > EPSILON)
		{
			this->scale(1.0 / mag);
		}
	}

	std::vector<double> as_vec()
	{
		return std::vector<double>{x, y};
	}

	std::unordered_map<std::string, double> as_umap()
	{
		std::unordered_map<std::string, double> ret;
		
		ret["x"] = x;
		ret["y"] = y;

		return ret;
	}
};


inline VecXY from_rtheta(double r, double theta)
{
	return VecXY(
		r * cos(theta),
		r * sin(theta)
	);

	// return v;
}

inline VecXY add_vecs(VecXY & a, VecXY & b) //addition operator overloaded function
{
	VecXY output(a);
	output.x += b.x;
	output.y += b.y;

	return output;
}

//addition operator overloaded function for vectors of positions
inline std::vector<VecXY> add_vecs(std::vector<VecXY> & a, std::vector<VecXY> & b) 
{
	assert(a.size() == b.size());
	std::vector<VecXY> output(a.size());

	for (size_t i = 0; i < a.size(); i++)
	{
		output[i] = add_vecs(a[i], b[i]);
	}

	return output;
}


std::pair<std::vector<double>, std::vector<size_t>> serialize(std::vector<VecXY> & positions)
{
	// init
	size_t n_particles = positions.size();
	std::vector<double> output_x(n_particles);
	std::vector<double> output_y(n_particles);
	std::vector<size_t> output_dims = { n_particles, 2 };

	// serialize
	for (size_t i = 0; i < n_particles; i++)
	{
		output_x[i] = positions[i].x;
		output_y[i] = positions[i].y;
	}

	// concatenate
	std::vector<double> output_data;
	output_data.reserve(2 * n_particles);
	output_data.insert( output_data.end(), output_x.begin(), output_x.end() );
	output_data.insert( output_data.end(), output_y.begin(), output_y.end() );

	return std::make_pair(output_data, output_dims);
}

std::pair<std::vector<double>, std::vector<size_t>> serialize(std::vector<std::vector<VecXY>> & position_steps)
{
	// check number of particles is constant
	size_t tsteps = position_steps.size();
	size_t n_particles = position_steps[0].size();
	for (size_t i = 1; i < tsteps; i++)
	{
		assert(position_steps[i].size() == n_particles);
	}

	// init
	std::vector<double> output_data;
	std::vector<size_t> output_dims = { tsteps, n_particles, 2 };

	// serialize
	for (size_t i = 0; i < tsteps; i++)
	{
		std::vector<double> tstep_data = serialize(position_steps[i]).first;
		output_data.insert( output_data.end(), tstep_data.begin(), tstep_data.end() );
	}

	// return
	return std::make_pair(output_data, output_dims);
}

VecXY get_displacement(VecXY a, VecXY b)
{
	return VecXY(
		a.x - b.x,
		a.y - b.y
	);
}

double dist(VecXY a, VecXY b)
{
	return pow((
		pow(a.x - b.x, 2.0)
		+ pow(a.y - b.y, 2.0)
	), 0.5);
}

double dist_sqrd(VecXY a, VecXY b)
{
	return (
		pow(a.x - b.x, 2.0)
		+ pow(a.y - b.y, 2.0)
	);
}



// func prototypes

// simple funcs
VecXY get_displacement(VecXY a, VecXY b);
double dist(VecXY a, VecXY b);
double dist_sqrd(VecXY a, VecXY b);

// the more complicated ones

std::vector<CollisionObject> load_objects(std::string collide_file)
{
	std::vector<CollisionObject> CollObjs = std::vector<CollisionObject>();

    // open file
    std::ifstream objfile(collide_file);
    if (!objfile.is_open() || !objfile.good())
    {
        exit(EXIT_FAILURE);
    }

    // initialize temp variables
	
	// stores type
	std::string raw_line;
	std::string str_colltype;

    // loop
    while(getline(objfile, raw_line))
	{
		std::istringstream liness(raw_line);
		liness >> str_colltype;

		if (str_colltype == "Box_Ax")
		{
			CollisionObject tempObj;
			tempObj.coll_type = Box_Ax;

			liness 
				>> tempObj.bound_min_x >> tempObj.bound_min_y 
				>> tempObj.bound_max_x >> tempObj.bound_max_y 
				>> tempObj.fvec_x >> tempObj.fvec_y;

			// store data
			CollObjs.push_back(tempObj);
		}
		else if (str_colltype == "Disc")
		{
			CollisionObject tempObj;
			tempObj.coll_type = Disc;

			liness 
				>> tempObj.bound_min_x >> tempObj.bound_min_y 
				>> tempObj.bound_max_x >> tempObj.bound_max_y 
				>> tempObj.centerpos_x >> tempObj.centerpos_y
				>> tempObj.force
				>> tempObj.radius_inner >> tempObj.radius_outer
				>> tempObj.angle_min >> tempObj.angle_max;

			// store data
			CollObjs.push_back(tempObj);
		}
    }

    // close file
    objfile.close();

	#ifdef COLLIDE_DEBUG
		for (CollisionObject obj : CollObjs)
		{
			std::cout << obj.coll_type << "," << obj.bound_max_x << "," << obj.bound_min_x << "," << obj.force << std::endl;
		}
	#endif

	return CollObjs;
}



void save_objects(std::string collide_file, std::vector<CollisionObject> & CollObjs)
{
    // open file
    std::ofstream objfile(collide_file);
    if (!objfile.is_open() || !objfile.good())
    {
        exit(EXIT_FAILURE);
    }

	// PRINTF_DEBUG("    >> elements in CollObjs vec: %ld\n", CollObjs.size())

	for (CollisionObject obj : CollObjs)
	{
		if (obj.coll_type == Box_Ax)
		{
			objfile 
				<< "Box_Ax"
				<< "\t" << obj.bound_min_x << "\t" << obj.bound_min_y 
				<< "\t" << obj.bound_max_x << "\t" << obj.bound_max_y 
				<< "\t" << obj.fvec_x << "\t" << obj.fvec_y
				<< std::endl;
		}
		else if (obj.coll_type == Disc)
		{
			objfile 
				<< "Disc"
				<< "\t" << obj.bound_min_x << "\t" << obj.bound_min_y 
				<< "\t" << obj.bound_max_x << "\t" << obj.bound_max_y 
				<< "\t" << obj.centerpos_x << "\t" << obj.centerpos_y
				<< "\t" << obj.force
				<< "\t" << obj.radius_inner << "\t" << obj.radius_outer
				<< "\t" << obj.angle_min << "\t" << obj.angle_max
				<< std::endl;
		}
		else
		{
			objfile << "NULL\n";
		}
    }

    // close file
	objfile.flush();
    objfile.close();
}



inline VecXY do_collide(CollisionObject & obj, VecXY pos)
{
	// forces on elements
	if (obj.coll_type == Box_Ax)
	{			
		if (
			(pos.x > obj.bound_min_x)
			&& (pos.x < obj.bound_max_x)
			&& (pos.y > obj.bound_min_y)
			&& (pos.y < obj.bound_max_y)
		){
			return VecXY(obj.fvec_x, obj.fvec_y);
		}
	}
	else if (obj.coll_type == Disc)
	{
		VecXY disc_center = VecXY(obj.centerpos_x, obj.centerpos_y);
		VecXY offset = get_displacement(pos, disc_center);
		double offset_mag = offset.mag();
		// compare to radius
		if (
			( offset_mag > obj.radius_inner) 
			&& (offset_mag < obj.radius_outer)
		)
		{
			// check wedge angles
			double angle = atan2(offset.y, offset.x);			
			if ( !( (obj.angle_min > angle) && (angle > obj.angle_max) ) )
			{
				// get the collision vector by normalizing and scaling offset
				offset.scale(obj.force / offset_mag);
				return offset;
			}
		}
	}

	// if no collisions found, return zero vec
	return VecXY();
}


// loop over all the objects and all the points
// and check for collisions

inline std::vector<VecXY> do_collide_vec(std::vector<VecXY> & pos_vec, std::vector<CollisionObject> & objs_vec)
{
	std::vector<VecXY> coll_vec;
	for (VecXY pos : pos_vec)
	{
		VecXY net_force = VecXY();
		for (CollisionObject obj : objs_vec)
		{
			VecXY obj_force = do_collide(obj, pos);
			net_force = add_vecs(net_force, obj_force);
		}
		coll_vec.push_back(net_force);
	}
	return coll_vec;
}


inline std::vector<VecXY> do_collide_vec_particles(std::vector<VecXY> & pos_vec, std::vector<CollisionObject> & objs_vec, double force_scalar)
{
	std::vector<VecXY> coll_vec;
	for (VecXY pos : pos_vec)
	{
		VecXY net_force = VecXY();
		for (CollisionObject obj : objs_vec)
		{
			VecXY obj_force = do_collide(obj, pos);
			obj_force.scale(force_scalar / obj_force.mag());
			net_force = add_vecs(net_force, obj_force);
		}
		coll_vec.push_back(net_force);
	}
	return coll_vec;
}



// TODO: do_collide_friction function






/* 
collision code originally from:
@article{Boyle_Berri_Cohen_2012, 
 	title={Gait Modulation in C. elegans: An Integrated Neuromechanical Model}, 
	volume={6}, 
	ISSN={1662-5188}, 
	url={https://www.frontiersin.org/articles/10.3389/fncom.2012.00010/full#h8}, 
 	DOI={10.3389/fncom.2012.00010}, 
	journal={Frontiers in Computational Neuroscience}, 
	publisher={Frontiers}, 
	author={Boyle, Jordan Hylke and Berri, Stefano and Cohen, Netta}, 
	year={2012}
modified by github.com/mivanit
} */
