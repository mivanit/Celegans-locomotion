// =============================================================
// Evolution of Integrated Neuromechanical Forward Locomotion
// Eduardo Izquierdo
// Indiana University
// February, 2018
// =============================================================

// #define EVOLVE
// #define PRINTTOFILE
// #define SEED
// #define OUTPUT
// #define SPEEDOUTPUT
// #define MAP_PHEN

#define ENABLE_CTOR_GENO 0
#define ENABLE_CTOR_PHENO 0
#define ENABLE_CTOR_JSON 1

#include "modules/output_macros.h"

#if ENABLE_CTOR_GENO
    #include "modules/TSearch.h"
    #include <pthread.h>
#endif

#include "modules/VectorMatrix.h"
#include "modules/Worm.h"
#include "modules/util.h"
#include "modules/FitnessCalc.h"
#include "consts.h"

#if ENABLE_CTOR_GENO
    #include "modules/evo_old.h"
#endif

#include <iostream>
#include <iomanip>  // cout precision
#include <algorithm>
#include <math.h>
#include <string>


using namespace std;


FitnessCalc EvaluationFunction(Worm w, RandomState &rs, double angle, std::vector<CollisionObject> & collObjs, string output_dir, double t_food_start)
{

    // open the files
    #ifdef _OUT_ANY
        PRINTF_DEBUG("  > opening output files in %s\n", output_dir.c_str())
    
        #ifdef _OUTW_FIT
            ofstream fitfile;
            fitfile.open(output_dir + "fitnes.yml");
        #endif

        #ifdef _OUTW_POS_ANY
            ofstream bodyfile;
            bodyfile.open(output_dir + "body.dat");
        #endif

        #ifdef _OUTW_ACT_ANY
            ofstream actfile;
            actfile.open(output_dir + "act.dat");
            w.DumpActState_header(actfile);
        #endif
        
        #ifdef _OUTW_CURVE
            ofstream curvfile;
            curvfile.open(output_dir + "curv.dat");

            PRINT_DEBUG("  > initializing curvature measurement\n")
            TVector<double> curvature(1, N_curvs);
            TVector<double> antpostcurv(1, 2);
            antpostcurv.FillContents(0.0);
        #endif

        // DEBUG: this tries to access something out of bounds. needs to be rewritten anyway to use json
        // PRINT_DEBUG("  > dumping worm params (NOT WORKING)\n")
        // w.DumpParams(paramsfile);

        // ofstream voltagefile;
        // ofstream paramsfile; 
        // paramsfile.open(output_dir + "params.dat");
    #endif



    #ifdef ENABLE_LEGACY_PARAMVEC
        PRINT_DEBUG("  > enabling legacy parameter vector\n")
        // this is disabled, dont use it.
        #if ENABLE_CTOR_GENO
            // Genotype-Phenotype Mapping
            TVector<double> phenotype(1, VectSize);
            GenPhenMapping(param_vec, phenotype);
            Worm w(phenotype, 0);
        #elif ENABLE_CTOR_PHENO
            Worm w(param_vec, 0);
        #endif
    #endif


    PRINT_DEBUG("  > initializing worm state\n")
    w.InitializeState(rs, angle, collObjs);

    PRINT_DEBUG("  > fitness calc init\n")
    FitnessCalc fcalc(w);

    // Time loop
    PRINT_DEBUG("  > starting time loop:\n\n")
    for (double t = 0.0; t <= DURATION; t += STEPSIZE) 
    {
        if (t > t_food_start)
        {
            w.chemo_re.enabled = true;
        }

        #ifdef UTIL_H_DEBUG
            // if on an integer step
            if ( (t - (int) t < STEPSIZE))
            {
                PRINTF_DEBUG("    >>  time:\t%f\t/\t%f\r", t, DURATION)
            }
        #endif

        // do the actual step
        w.Step(STEPSIZE, 1);

        // update fitness
        fcalc.update();

        // dump states to files
        #ifdef _OUT_ANY
                #ifdef _OUTW_CURVE
                    w.Curvature(curvature);
                    curvfile << curvature << endl;
                #endif

                #ifdef _OUTW_POS_ANY
                    w.DumpBodyState(bodyfile, skip);
                #endif

                #ifdef _OUTW_ACT_ANY
                    w.DumpActState(actfile, skip);
                #endif
        #endif
    }

    PRINT_DEBUG("\n\n  > finished time loop!\n")

    // close the files
    #ifdef _OUT_ANY
        PRINTF_DEBUG("  > closing files, saving to %s\n", output_dir.c_str())

        #ifdef _OUTW_POS_ANY
            bodyfile.close();
        #endif

        #ifdef _OUTW_ACT_ANY
            actfile.close();
        #endif

        #ifdef _OUTW_CURVE
            curvfile.close();
        #endif

        #ifdef _OUTW_FIT
            fitfile << fcalc.strprintf();
        #endif
    #endif

    // print fitness to console
    PRINT_DEBUG("\n\n    > fitnesses:\n")
    std::string fcalc_output = fcalc.strprintf();
    std::replace(fcalc_output.begin(), fcalc_output.end(), '\n', '\t');
    PRINT_DEBUG(fcalc_output.c_str())

    return fcalc;
}

