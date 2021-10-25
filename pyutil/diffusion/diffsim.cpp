#include <random>
#include <vector>

#include "Collide_standalone.h"
#include "../../modules/packages/cxxopts.hpp"
#include "../../modules/packages/npy.hpp"

#include "../../modules/packages/json.hpp"
using json = nlohmann::json;
// using vector = std::vector;

const double PI = 3.14159265358979323846;

// NOTE: this gets overwritten by the command line arguments
// diffusion factor is nan for this reason
double diffusion_factor = std::nan("");

static std::default_random_engine generator;
static std::uniform_real_distribution dist_angle(0.0, 2.0*PI);
// TODO: fix this bit with the randomized travel distance
// static std::exponential_distribution<double> dist_exponential(diffusion_factor);

std::vector<CollisionObject> COLL_OBJS;


inline std::vector<double> get_angle(int size)
{
	std::vector<double> vec(size);
    std::generate(vec.begin(), vec.end(), [&]{ 
		return dist_angle(generator);
	});
	return vec;
}

inline std::vector<double> get_traveldist(int size)
{
	// TODO: this isnt working properly, disabled
	std::vector<double> vec(size);
	// std::generate(vec.begin(), vec.end(), [&]{
	// 	return dist_exponential(generator);
	// });
	return vec;
}

std::vector<VecXY> initialize_particles(VecXY pos, int size)
{
	std::vector<VecXY> vec(size, pos);
	return vec;
}

std::vector<VecXY> iterate_particles(std::vector<VecXY> positions)
{
	// update positions
	std::vector<VecXY> newpos(positions.size());

	std::vector<double> pd_angles = get_angle(positions.size());
	// std::vector<double> pd_distances = get_traveldist(positions.size());
		
	for (int i = 0; i < positions.size(); i++)
	{
		// VecXY pos_delts = from_rtheta(pd_distances[i], pd_angles[i]);
		VecXY pos_delts = from_rtheta(diffusion_factor, pd_angles[i]);
		newpos[i] = add_vecs(positions[i], pos_delts);
	}

	// do collisions
	std::vector<VecXY> coll_delta = do_collide_vec_particles(newpos, COLL_OBJS);
	newpos = add_vecs(newpos, coll_delta);

	return newpos;
}


std::pair<
	std::vector<std::vector<VecXY>>, 
	std::vector<int>
> do_sim(
	VecXY pos, 
	long unsigned size, 
	long unsigned tsteps,
	long int save_every,
	int print_every = 100
){
	// pos_store will be returned
	std::vector<std::vector<VecXY>> pos_store;
	pos_store.reserve(tsteps/save_every);
	std::vector<int> tstep_store;

	// pos_current will store the current state only
	std::vector<VecXY> pos_current = initialize_particles(pos, size);
	std::cout << std::endl;
	for (long unsigned i = 0; i < tsteps+1; i++)
	{
		pos_current = iterate_particles(pos_current);
		if (i % print_every == 0)
		{
			std::cout << "> iteration\t" << i << "\r";
			std::cout.flush();
		}
		if (
			( (i % save_every == 0) && (i > 0) )
			|| (i == tsteps)
		){
			pos_store.push_back(pos_current);
			tstep_store.push_back(i);
		}
	}
	std::cout << std::endl;
	
	return std::make_pair(pos_store, tstep_store);
}



VecXY get_foodPos(cxxopts::ParseResult & cmd)
{
    // get food position

	std::string str_foodPos = cmd["foodPos"].as<std::string>();

	int idx_comma = str_foodPos.find(',');
	double foodPos_x = std::stod(str_foodPos.substr(0,idx_comma));
	double foodPos_y = std::stod(str_foodPos.substr(idx_comma+1, std::string::npos));

	return VecXY(foodPos_x, foodPos_y);
}


int main (int argc, const char* argv[])
{
    // set up command line parser
    cxxopts::Options options("diffsim.cpp", "particle diffusion sim with collisions");
    options.add_options()
        ("c,coll", "collision tsv file", 
            cxxopts::value<std::string>()->default_value("../../input/objs/maze.tsv"))
        ("o,output", "output file", 
            cxxopts::value<std::string>())
        ("d,duration", "sim duration in timeteps", 
            cxxopts::value<long unsigned>()->default_value("1000"))
		("n,nparticles", "number of particles", 
            cxxopts::value<long unsigned>()->default_value("1000"))
        ("f,foodPos", "food position (comma separated)", 
            cxxopts::value<std::string>()->default_value("0,0"))
		("s,save_every", "num timesteps between saving. -1 (default) implies only final step saved", 
			cxxopts::value<long int>()->default_value("-1"))
		("a,diffusion_factor", "diffusion step size",
			cxxopts::value<double>()->default_value("0.001"))
        ("h,help", "print usage")
    ;

    // read command, handle help printing
    cxxopts::ParseResult cmd = options.parse(argc, argv);
    if (cmd.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

	// get parameters
    long unsigned duration = cmd["duration"].as<long unsigned>();
	long unsigned nparticles = cmd["nparticles"].as<long unsigned>();
	diffusion_factor = cmd["diffusion_factor"].as<double>();
	
	long int save_every = cmd["save_every"].as<long int>();
	if (save_every == -1) { save_every = duration; }

	std::string output_basename = cmd["output"].as<std::string>();
	std::string collision_file = cmd["coll"].as<std::string>();
	VecXY foodPos = get_foodPos(cmd);

	std::cout << "read parameters:"
		<< "\n\tduration:\t" << duration
		<< "\n\tnparticles:\t" << nparticles
		<< "\n\toutput_basename:\t" << output_basename
		<< "\n\tcollision_file:\t" << collision_file
		<< "\n\tfoodPos:\t" << foodPos.x << ", " << foodPos.y
		<< std::endl;

	json metadata = {
		{"duration", duration},
		{"nparticles", nparticles},
		{"output_basename", output_basename},
		{"collision_file", collision_file},
		{"foodPos", foodPos.as_umap()},
		{"diffusion_factor", diffusion_factor},
		{"save_every", save_every} 
	};


	// load collision objects
	COLL_OBJS = load_objects(collision_file);
	std::vector<std::unordered_map<std::string, double>> coll_objs_map;
	coll_objs_map.reserve(COLL_OBJS.size());
	for (CollisionObject obj : COLL_OBJS)
	{
		coll_objs_map.push_back(obj.as_umap());
	}

	std::cout << "\tobjects_count:\t" << COLL_OBJS.size() << std::endl;

	metadata["collision_data"] = coll_objs_map;
	
	// run simulation
	std::pair<
		std::vector<std::vector<VecXY>>, 
		std::vector<int>
	> positions = do_sim(
		foodPos,
		nparticles,
		duration,
		save_every
	);

	std::cout << "\n\nsim complete!" << std::endl;

	metadata["tsteps"] = positions.second;

	// save metadata
	std::ofstream ofs(output_basename + ".json");
	ofs << std::setw(4) << metadata << std::endl;
	ofs.flush(); ofs.close();

	// save data in numpy format
	std::pair<std::vector<double>, std::vector<size_t>> data = serialize(positions.first);
	
	// NOTE: new modified interface -- NOT standard
	npy::SaveArrayAsNumpy(
		output_basename + ".npy", // filename
		false, // fortran_order
		data.second, // shape
		data.first // data
	);

	// old interface
	// npy::SaveArrayAsNumpy(
	// 	output_basename, // filename
	// 	false, // fortran_order
	// 	(unsigned int) data.second.size(), // n_dims
	// 	(unsigned long *) data.second.data(), // shape
	// 	data.first // data
	// );
}