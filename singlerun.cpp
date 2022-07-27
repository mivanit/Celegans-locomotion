#define COLLIDE

#define ENABLE_CTOR_JSON 1
#include <iostream>
using namespace std;
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
#include "modules/FitnessCalc.h"

#include "main.h"

// #include <filesystem>


// function prototyping
template <typename T_cast_cmd>
inline void overwrite_json_from_cmd(
    json & simulation_params,
    cxxopts::ParseResult & cmd,
    string field
){
    if (cmd.count(field))
    {
        simulation_params[field] = cmd[field].as<T_cast_cmd>();
    }
}

inline void copy_config_files(
    std::string output_dir,
    json & params,
    std::vector<CollisionObject> & collObjs
);

inline int set_seed(json & simulation_params, cxxopts::ParseResult & cmd);

inline void set_foodPos(json & params, cxxopts::ParseResult & cmd);



/*

 #    #   ##   # #    #
 ##  ##  #  #  # ##   #
 # ## # #    # # # #  #
 #    # ###### # #  # #
 #    # #    # # #   ##
 #    # #    # # #    #

*/


int main (int argc, const char* argv[])
{
    // set output precision
    std::cout << std::setprecision(10);

    // ========================================
    // set up command line parser
    // ========================================
    cxxopts::Options options("PhysWormSim", "Mechanical and electrophysiological simulation of C. elegans nematode");
    options.add_options()
        ("p,params", "params json file", 
            cxxopts::value<std::string>()->default_value("input/params.json"))
        ("c,coll", "collision tsv file", 
            cxxopts::value<std::string>())
        ("a,angle", "starting angle", 
            cxxopts::value<double>())
        ("o,output", "output dir", 
            cxxopts::value<string>())
        ("r,rand", "random initialization seed based on time", 
            cxxopts::value<bool>())
        ("d,duration", "sim duration in seconds", 
            cxxopts::value<double>())
        ("f,foodPos", "food position (comma separated). set to \"DISABLE\" to set the scalar to zero", 
            cxxopts::value<string>())
        ("s,seed", "set random initialization seed. takes priority over `rand`. seed is 0 by default.", 
            cxxopts::value<long>())
        ("h,help", "print usage")
    ;

    // read command, handle help printing
    cxxopts::ParseResult cmd = options.parse(argc, argv);
    if (cmd.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);    
    }

    // ========================================
    // loading json
    // ========================================
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
    params["simulation"]["src-params"] = cmd["params"].as<std::string>();


    // ========================================
    PRINT_DEBUG("  > read command line args\n")
    // ========================================

    json simulation_params = params["simulation"];

    // get random seed
    long seed = set_seed(simulation_params, cmd);
    RandomState rs;
    rs.SetRandomSeed(seed);
    PRINTF_DEBUG("  >> set rand seed to %ld\n", seed)

    // set duration
    overwrite_json_from_cmd<double>(simulation_params, cmd, "duration");
    DURATION = simulation_params["duration"].get<double>();
    PRINTF_DEBUG("  >> set duration to %f\n", DURATION)

    // set angle
    overwrite_json_from_cmd<double>(simulation_params, cmd, "angle");
    PRINTF_DEBUG("  >> set angle to %f\n", params["simulation"]["angle"].get<double>())

    // set food position
    set_foodPos(params, cmd);
    PRINTF_DEBUG(
        "  >> set foodPos to %f, %f\n", 
        params["ChemoReceptors"]["foodPos"]["x"].get<double>(),
        params["ChemoReceptors"]["foodPos"]["y"].get<double>()
    )
    

    // set collision objects
    overwrite_json_from_cmd<std::string>(simulation_params, cmd, "coll");
    std::vector<CollisionObject> collObjs = load_objects(simulation_params["coll"].get<std::string>());
    PRINTF_DEBUG("  >> collision tsv from:\t%s\n", simulation_params["coll"].get<std::string>().c_str())

    // set output dir
    overwrite_json_from_cmd<string>(simulation_params, cmd, "output");

    // REVIEW: is this required? not sure
    params["simulation"] = simulation_params;

    // copy configs
    copy_config_files(simulation_params["output"].get<std::string>(), params, collObjs);

    // ========================================
    // setting up simulation
    // ========================================

    InitializeBodyConstants();
    PRINT_DEBUG("> finished init body constants\n")


    PRINT_DEBUG("> creating worm object:\n")
    Worm wrm(params);

    // ========================================
    // run simulation
    // ========================================
    PRINT_DEBUG("> running evaluation:\n")
    FitnessCalc fcalc = EvaluationFunction(
        wrm, 
        rs,
        params["simulation"]["angle"].get<double>(),
        collObjs,
        params["simulation"]["output"].get<std::string>(),
        params["simulation"]["t_food_start"].get<double>()
    );
    
    PRINT_DEBUG("> finished!\n")
    return 0;
}



inline void copy_config_files(
    std::string output_dir,
    json & params,
    std::vector<CollisionObject> & collObjs
){
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



/*

 #    # ###### #      #####  ###### #####   ####
 #    # #      #      #    # #      #    # #
 ###### #####  #      #    # #####  #    #  ####
 #    # #      #      #####  #      #####       #
 #    # #      #      #      #      #   #  #    #
 #    # ###### ###### #      ###### #    #  ####

*/

/*
gets the seed (possibly random) from the command line option, overwriting whatever is in params.json

RETURNS: seed value
MODIFIES: `json simulation_params`
*/
inline int set_seed(json & simulation_params, cxxopts::ParseResult & cmd)
{
    long seed;
    if (cmd.count("seed"))
    {
        seed = cmd["seed"].as<long>();
    }
    else if (cmd.count("rand"))
    {
        seed = static_cast<long>(time(NULL));
    }
    else
    {
        seed = simulation_params["seed"].get<long>();
    }

    simulation_params["seed"] = seed;
    return seed;
}



inline void set_foodPos(json & params, cxxopts::ParseResult & cmd)
{
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
                    params["ChemoReceptors"]["kappa"] = 0.0;
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
}









// // storing meta params in json file
// if (params.contains("simulation"))
// {
//     fprintf(stderr, "\n\nWARNING: input json does not contain 'simulation' section -- the code will probably break")
//     params["simulation"] = {
//         {"duration", DURATION},
//         {"seed", seed},
//         {"angle", cmd["angle"].as<double>()},
//         {"src-params", cmd["params"].as<std::string>()},
//         {"src-coll", cmd["coll"].as<std::string>()},
//         {"output",  cmd["output"].as<std::string>()}
//     };
// }
