#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>


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

};

// x-y vector stuff
struct VecXY
{
	double x;
	double y;

	VecXY(double in_x, double in_y)
	{
		x = in_x;
		y = in_y;
	}

	VecXY(VecXY & in_vec)
	{
		x = in_vec.x;
		y = in_vec.y;
	}

	VecXY()
	{
		x = 0.0;
		y = 0.0;
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
};

VecXY add_vecs(VecXY & a, VecXY & b) //addition operator overloaded function
{
	VecXY output(a);
	output.x += b.x;
	output.y += b.y;

	return output;
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

	PRINTF_DEBUG("    >> elements in CollObjs vec: %ld\n", CollObjs.size())

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

inline std::vector<VecXY> do_collide_vec(inline std::vector<VecXY> & pos_vec, std::vector<CollisionObject> & objs_vec)
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
