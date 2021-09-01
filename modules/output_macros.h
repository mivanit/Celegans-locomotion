#ifndef H_OUTPUT_MACROS
#define H_OUTPUT_MACROS

// NOTE: checking for only one output flag passed is done only if _OUT_NONE is passed

// TODO: checking for definitions is prone to bugs, 
// better to check for the value of the flag so that 
// compilation fails if we check for a flag that is not defined

#ifdef _OUT_NONE

    #pragma message "enabling no output mode `_OUT_NONE`"

    #ifdef _OUT_MIN
        #error both `_OUT_NONE` and `_OUT_MIN` passed, expect only one
    #endif
    #ifdef _OUT_SHORT
        #error both `_OUT_NONE` and `_OUT_SHORT` passed, expect only one
    #endif
    #ifdef _OUT_FULL
        #error both `_OUT_NONE` and `_OUT_FULL` passed, expect only one
    #endif

    #define FOUND_OUTMODE
#endif

#ifdef _OUT_MIN
    #pragma message "enabling minimal output mode `_OUT_MIN`"

    #define _OUT_ANY
    
    #define _OUTW_POS_ANY
    #define _OUTW_POS_HEAD

    #define FOUND_OUTMODE
#endif

#ifdef _OUT_SHORT
    #pragma message "enabling short output mode `_OUT_SHORT`"

    #define _OUT_ANY

    #define _OUTW_POS_ANY
    #define _OUTW_POS_HEAD

    #define _OUTW_ACT_ANY
    #define _OUTW_ACT_HEAD

    #define FOUND_OUTMODE
#endif


// if none of the shortened output modes specified, do a full output
#ifndef FOUND_OUTMODE
    #define _OUT_FULL
#endif

#ifdef _OUT_FULL
    #pragma message "enabling full output mode `_OUT_FULL`"

    #define _OUT_ANY
    
    #define _OUTW_POS_ANY
    #define _OUTW_POS_HEAD
    #define _OUTW_POS_BODY
    
    #define _OUTW_ACT_ANY
    #define _OUTW_ACT_VC
    #define _OUTW_ACT_SR
    #define _OUTW_ACT_MUSC

    #define _OUTW_VOLT_ANY
    #define _OUTW_VOLT_HEAD
    #define _OUTW_VOLT_BODY

    #define _OUTW_FIT
    
    #define _OUTW_CURVE
#endif

#endif