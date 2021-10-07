#include <random>
#include <vector>

#include "Collide_standalone.h"
#include "../../modules/packages/cxxopts.hpp"
#include "../../modules/packages/npy.hpp"


const double PI = 3.14159265358979323846;
const double TRAVELDIST_LAMBDA = 0.00001;

static std::default_random_engine generator;
static std::uniform_real_distribution dist_angle(0.0, 2.0*PI);
static std::exponential_distribution<double> dist_exponential(TRAVELDIST_LAMBDA);

std::vector<CollisionObject> COLL_OBJS;


std::vector<double> get_angle(int size)
{
	std::vector<double> vec(size);
    std::generate(vec.begin(), vec.end(), [&]{ return dist_angle(generator);});
	return vec;
}

std::vector<double> get_traveldist(int size)
{
	std::vector<double> vec(size);
	std::generate(vec.begin(), vec.end(), [&]{ return dist_exponential(generator)});
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
	{
		std::vector<double> pd_angles = get_angle(positions.size());
		std::vector<double> pd_distances = get_traveldist(positions.size());
		
		for (int i = 0; i < positions.size(); i++)
		{
			newpos[i] = add_vecs(
				positions[i], 
				from_rtheta(pd_distances[i], pd_angles[i]
			);
		}
	}

	// do collisions
	newpos = do_collide_vec(newpos, COLL_OBJS);

	return newpos;
}


std::vector<VecXY> do_sim(VecXY pos, long unsigned size, long unsigned tsteps)
{
	std::vector<VecXY> positions = initialize_particles(pos, size);
	for (long unsigned i = 0; i < tsteps; i++)
	{
		positions = iterate_particles(positions);
	}
	return positions;
}



VecXY get_foodPos(cxxopts::ParseResult & cmd)
{
    // get food position

	string str_foodpos = cmd["foodPos"].as<std::string>();

	int idx_comma = str_foodpos.find(',');
	double foodpos_x = std::stod(str_foodpos.substr(0,idx_comma));
	double foodpos_y = std::stod(str_foodpos.substr(idx_comma+1, std::string::npos));

	return VecXY(foodpos_x, foodpos_y);
}


int main (int argc, const char* argv[])
{
    // set up command line parser
    cxxopts::Options options("diffsim.cpp", "particle diffusion sim with collisions");
    options.add_options()
        ("c,coll", "collision tsv file", 
            cxxopts::value<std::string>())
        ("o,output", "output file", 
            cxxopts::value<string>())
        ("d,duration", "sim duration in timeteps", 
            cxxopts::value<long unsigned>())
		("n,nparticles", "number of particles", 
            cxxopts::value<long unsigned>())
        ("f,foodPos", "food position (comma separated)", 
            cxxopts::value<string>())
        // ("r,rand", "random initialization seed based on time", 
        //     cxxopts::value<bool>())
        // ("s,seed", "set random initialization seed. takes priority over `rand`. seed is 0 by default.", 
        //     cxxopts::value<long>())
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
	string output_file = cmd["output"].as<string>();
	string collision_file = cmd["coll"].as<string>();
	VecXY foodpos = get_foodPos(cmd);

	// load collision objects
	COLL_OBJS = load_objects(collision_file);

	// run simulation
	std::vector<VecXY> final_positions = do_sim(
		foodpos,
		nparticles,
		duration
	);

	// save data in numpy format
	std::pair<std::vector<double>, std::vector<size_t>> data = serialize(final_positions);
	
	npt::SaveArrayAsNumpy(
		output_file, // filename
		false, // fortran_order
		data.second.size(), // n_dims
		data.second.data(), // shape
		data.first // data
	);
}



int test_save(void) {
  const long unsigned leshape [] = {2,3};
  vector<double> data {1, 2, 3, 4, 5, 6};
  npy::SaveArrayAsNumpy("data/out.npy", false, 2, leshape, data);

  const long unsigned leshape2 [] = {6};
  npy::SaveArrayAsNumpy("data/out2.npy", false, 1, leshape2, data);

  return 0;
}