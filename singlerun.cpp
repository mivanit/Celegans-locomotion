#define PRINTTOFILE
//#define SEED

#define OUTPUT
#define SPEEDOUTPUT
#define COLLIDE

#define ENABLE_CTOR_JSON 1

#ifdef _WIN32
    #include <direct.h>
    #define MKDIR _mkdir
#elif defined __linux__
    #include <sys/stat.h>
    #define MKDIR(path) \
        mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

#include "modules/packages/cxxopts.hpp"
#include "modules/util.h"
#include "modules/Collide.h"

#include "main.h"

// #include <filesystem>

int main (int argc, const char* argv[])
{
    // set output precision
    std::cout << std::setprecision(10);

    // set up command line parser
    cxxopts::Options options("PhysWormSim", "Mechanical and electrophysiological simulation of C. elegans nematode");
    options.add_options()
        ("p,params", "params json file", 
            cxxopts::value<std::string>()->default_value("input/params.json"))
        ("c,coll", "collision tsv file", 
            cxxopts::value<std::string>()->default_value("input/collision_objs.tsv"))
        ("a,angle", "starting angle", 
            cxxopts::value<double>()->default_value("1.570795"))
        ("o,output", "output dir", 
            cxxopts::value<string>()->default_value("data/run/"))
        ("r,rand", "random initialization seed based on time", 
            cxxopts::value<bool>())
        ("d,duration", "sim duration in seconds", 
            cxxopts::value<double>()->default_value("100.0"))
        ("f,foodPos", "food position (comma separated) (defaults to whatever is in params.json). set to \"DISABLE\" to set the scalar to zero", 
            cxxopts::value<string>())
        ("s,seed", "set random initialization seed. takes priority over `rand`. seed is 0 by default.", 
            cxxopts::value<long>())
        ("h,help", "print usage")
    ;

    // read command, handle help printing
    auto cmd = options.parse(argc, argv);

    if (cmd.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    
    }
    PRINT_DEBUG("> read command line args\n")
    
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
    PRINTF_DEBUG("> set rand seed to %d\n", seed)

    // set duration
    DURATION = cmd["duration"].as<double>();;


    // setting up simulation
    InitializeBodyConstants();
    PRINT_DEBUG("> finished init body constants\n")
    
    PRINT_DEBUG("> loading configs:\n")
    PRINTF_DEBUG("  > params json from:  \t%s\n", cmd["params"].as<std::string>().c_str())
    std::ifstream ifs(cmd["params"].as<std::string>());
    json params = json::parse(
        std::string(
            (std::istreambuf_iterator<char>(ifs) ),
            (std::istreambuf_iterator<char>()    )
        ),
        nullptr,
        true,
        true
    );


    // get food position and (maybe) disable chemosensation
    if (cmd.count("foodPos"))
    { 
        if (params.contains("ChemoReceptors"))
        {
            {
                string str_foodpos = cmd["foodPos"].as<std::string>();
                PRINTF_DEBUG("    > loading food pos from string: %s", str_foodpos.c_str())

                if (str_foodpos == "DISABLE")
                {
                    // if food sensation is disabled, note that in the json and disable input
                    params["ChemoReceptors"]["stim_scalar"] = 0.0;
                    params["ChemoReceptors"]["DISABLED"] = true;

                    params["ChemoReceptors"]["foodPos"]["x"] = nan("");
                    params["ChemoReceptors"]["foodPos"]["y"] = nan("");
                }
                else
                {
                    int idx_comma = str_foodpos.find(',');
                    double foodpos_x = std::stod(str_foodpos.substr(0,idx_comma));
                    double foodpos_y = std::stod(str_foodpos.substr(idx_comma+1, std::string::npos));

                    params["ChemoReceptors"]["foodPos"]["x"] = foodpos_x;
                    params["ChemoReceptors"]["foodPos"]["y"] = foodpos_y;
                }
            }
        }
        else
        {
            throw std::runtime_error("foodPos given, but \"ChemoReceptors\" not enabled in params.json");
        }
    }
    



    PRINTF_DEBUG("  > collision tsv from:\t%s\n", cmd["coll"].as<std::string>().c_str())
    std::vector<CollisionObject> collObjs = load_objects(cmd["coll"].as<std::string>());

    // copy configs
    {
        std::string output_dir = cmd["output"].as<std::string>();
        PRINTF_DEBUG("> creating output dir: %s\n", output_dir.c_str())
        MKDIR(output_dir.c_str());

        std::string outpath_collobjs = output_dir + "coll_objs.tsv";
        std::string outpath_params = output_dir + "params.json";

        PRINTF_DEBUG("  > copying collision objects to:\t%s\n", outpath_collobjs.c_str())
        save_objects(outpath_collobjs, collObjs);
        
        PRINTF_DEBUG("  > copying params json to:      \t%s\n", outpath_params.c_str())
        std::ofstream ofs_params(outpath_params);
        ofs_params << params.dump(1, '\t');
        ofs_params.flush();
        ofs_params.close();
    }

    PRINT_DEBUG("> creating worm object:\n")
    Worm wrm(params);


    PRINT_DEBUG("> running evaluation:\n")
    EvaluationFunction(
        wrm, 
        rs,
        cmd["angle"].as<double>(),
        collObjs,
        cmd["output"].as<std::string>()
    );
    
    // TODO: make this happen after worm init but before sim?
    PRINT_DEBUG("> saving copies of input params\n")

    PRINT_DEBUG("> finished!\n")
    return 0;
}

