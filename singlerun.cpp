#define PRINTTOFILE
//#define SEED
#define OUTPUT
#define SPEEDOUTPUT
#define COLLIDE

#define ENABLE_CTOR_JSON 1

#include "modules/packages/cxxopts.hpp"

#include "main.h"

int main (int argc, const char* argv[])
{
    // set output precision
    std::cout << std::setprecision(10);

    // set up command line parser
    cxxopts::Options options("PhysWormSim", "Mechanical and electrophysiological simulation of C. elegans nematode");
    options.add_options()
        ("p,params", "params json file", cxxopts::value<std::string>()->default_value("input/params.json"))
        ("c,coll", "collision tsv file", cxxopts::value<std::string>()->default_value("input/collision_objs.tsv"))
        ("a,angle", "starting angle", cxxopts::value<double>()->default_value("1.570795"))
        ("o,output", "output dir", cxxopts::value<string>()->default_value("data/run/"))
        ("r,rand", "random initialization seed based on time", cxxopts::value<bool>())
        ("s,seed", "set random initialization seed. takes priority over `rand`. seed is 0 by default.", cxxopts::value<long>())
        ("h,help", "print usage")
    ;

    // read command, handle help printing
    auto cmd = options.parse(argc, argv);

    if (cmd.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }
    
    // get random seed
    RandomState rs;
    long seed = 0;
    if (cmd.count("seed"))
    {
        seed = cmd["seed"].as<long>();
    }
    else if (cmd.count("rand"))
    {
        seed = static_cast<long>(time(NULL));
    }
    rs.SetRandomSeed(seed);



    // setting up simulation
    InitializeBodyConstants();
    // load worm
    std::ifstream ifs(cmd["params"].as<std::string>());
    json params = json::parse(std::string(
        (std::istreambuf_iterator<char>(ifs) ),
        (std::istreambuf_iterator<char>()    ) 
    ));
    Worm wrm(params);

    EvaluationFunction(
        wrm, 
        rs, 
        cmd["angle"].as<double>(),
        cmd["coll"].as<std::string>(),
        cmd["output"].as<std::string>()
    );
    return 0;
}
