#ifndef H_FITNESSCALC
#define H_FITNESSCALC

#include <string>

#include "../consts.h"
#include "Worm.h"

class FitnessCalc
{
public:

// position measures
double xt,yt;
double xtp,ytp;

// fitness measures
double bodyorientation;
double movementorientation;
double anglediff;
double distancetravelled;

// temp vars
double temp_fwd;

// reference to the worm we care about
Worm & w;

FitnessCalc(Worm & in_w) : w(in_w)
{
	xt = w.CoMx();
    yt = w.CoMy();
}

bool update()
{

	// Current and past centroid position
	xtp = xt; ytp = yt;
	xt = w.CoMx(); yt = w.CoMy();

	// Integration error check
	if (
		isnan(xt) 
		|| isnan(yt) 
		|| sqrt(pow(xt-xtp,2)+pow(yt-ytp,2)) > 100*AvgSpeed*STEPSIZE
	){
		std::cerr << "NaN postion!" << std::endl;
		return true;
	}

	// Fitness
	
	// Orientation of the body position
	bodyorientation = w.Orientation();                  
	
	// Orientation of the movement
	movementorientation = atan2(yt-ytp,xt-xtp);         
	
	// Check how orientations align
	anglediff = movementorientation - bodyorientation;  
	
	// Add to fitness only movement forward
	temp_fwd = cos(anglediff) > 0.0 ? 1.0 : -1.0;           
	distancetravelled += temp_fwd * sqrt(pow(xt-xtp,2)+pow(yt-ytp,2));

	return false;
}

string strprintf()
{
	char buffer [512];
	sprintf(
		buffer,
		"bodyorientation: %f\nmovementorientation: %f\nanglediff: %f\ndistancetravelled: %f\n",
		bodyorientation,
		movementorientation,
		anglediff,
		distancetravelled
	);

	return string(buffer);
}

};


#endif