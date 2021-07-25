from typing import *

__EXPECTED_PATH__ : str = 'pyutil.geno_distr'
if not (TYPE_CHECKING or (__name__ == __EXPECTED_PATH__)):
	sys.path.append(os.path.join(
		sys.path[0], 
		'../' * __EXPECTED_PATH__.count('.'),
	))

from pyutil.util import *
from pyutil.params import *

ranges_chemo_v6 : ModParamsRanges = {
	ModParam("conn",   "Head,AWA,RIM,chem") : RangeTuple(-40000,40000),
	ModParam("conn",   "Head,RIM,RMD*,chem") : RangeTuple(-1000,1000),
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(0.0, 1000.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-100000, -1000000),
	ModParam("params", "Head.neurons.AWA.tau") : RangeTuple(0.0, 0.5),
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-10.0, 10.0),
	ModParam("params", "Head.neurons.RIM.tau") : RangeTuple(0.0, 0.5),
	ModParam("params", "Head.neurons.RIM.theta") : RangeTuple(-10.0, 10.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}


# desmos reference
# https://www.desmos.com/calculator/wqyexesoml
# better one:
# https://www.desmos.com/calculator/c4rr96zrxc
ranges_chemo_v7 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-20.0, 20.0),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-600.0, 100.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-100.0,100.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(150.0, 300.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-50000, -10000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(0.0, 5.0),
	ModParam("params", "Head.neurons.RIA.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(0.0, 1.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}


ranges_chemo_v7_1 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-50.0, 20.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-1.0, 1.0),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-600.0, 600.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-100.0, 100.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(150.0, 300.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-50000, -10000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(-5.0, 5.0),
	ModParam("params", "Head.neurons.RIA.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-1.0, 1.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

ranges_chemo_v7_2 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-50.0, 20.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-1.0, 1.0),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-600.0, 600.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-100.0, 100.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(150.0, 300.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-50000, -10000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(-5.0, 5.0),
	ModParam("params", "Head.neurons.RIA.tau") : RangeTuple(0.005, 0.01),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-5.0, 5.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

ranges_chemo_v8_1 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-30.0, 10.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-0.1, 0.1),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-20.0, 20.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-10.0, 10.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(150.0, 300.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-50000, -10000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(-5.0, 5.0),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-5.0, 5.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

ranges_chemo_v8_2 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-50.0, 10.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-0.05, 0.05),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-20.0, 20.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-10.0, 10.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(150.0, 300.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-50000, -10000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(-5.0, 5.0),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-5.0, 5.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}


ranges_chemo_v9_1 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-50.0, -10.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-0.05, 0.05),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-40.0, 20.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-15.0, 10.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(200.0, 250.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-60000, -40000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 0.0),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(0.0, 5.0),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-5.0, 0.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

ranges_chemo_v9_2 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-50.0, 10.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-0.05, 0.05),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-40.0, 20.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-15.0, 10.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(200.0, 250.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-60000, -40000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(0.0, 5.0),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-5.0, 0.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

ranges_chemo_v_idk : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-50.0, 10.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-0.05, 0.05),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-40.0, 20.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-15.0, 10.0),
	ModParam("conn",   "Head,SMD*,RIA,chem") : RangeTuple(-0.1, 0.1),
	ModParam("conn",   "Head,RMD*,RIA,chem") : RangeTuple(-0.1, 0.1),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(200.0, 250.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-60000, -40000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(0.0, 5.0),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-5.0, 0.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

ranges_chemo_v12_1 : ModParamsRanges = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : RangeTuple(-50.0, 10.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : RangeTuple(-1.0, 1.0),
	ModParam("conn",   "Head,AIY,RIA,chem") : RangeTuple(-50.0, 20.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : RangeTuple(-25.0, 10.0),
	ModParam("conn",   "Head,SMD*,RIA,chem") : RangeTuple(-3.0, 3.0),
	ModParam("conn",   "Head,RMD*,RIA,chem") : RangeTuple(-3.0, 3.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : RangeTuple(180.0, 270.0),
	ModParam("params", "ChemoReceptors.lambda") : RangeTuple(-65000, -40000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : RangeTuple(-1.0, 1.0),
	ModParam("params", "Head.neurons.AIY.theta") : RangeTuple(0.0, 5.0),
	ModParam("params", "Head.neurons.RIA.theta") : RangeTuple(-5.0, 0.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

dists_chemo_v14_1 : ModParamsDists = {
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : NormalDistTuple(mu=-11.7, sigma=5.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : NormalDistTuple(mu=-2.14, sigma=1.0),
	ModParam("conn",   "Head,AIY,RIA,chem") : NormalDistTuple(mu=-34.5, sigma=10.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : NormalDistTuple(mu=-13.1, sigma=5.0),
	ModParam("conn",   "Head,RIA,SMD*,chem") : NormalDistTuple(mu=0.0, sigma=5.0),
	ModParam("conn",   "Head,RIA,RIA,chem") : NormalDistTuple(mu=0.0, sigma=0.1),
	ModParam("conn",   "Head,SMD*,RIA,chem") : NormalDistTuple(mu=0.0, sigma=1.0),
	ModParam("conn",   "Head,RMD*,RIA,chem") : NormalDistTuple(mu=0.0, sigma=1.0),
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : NormalDistTuple(mu=209.5, sigma=30.0),
	ModParam("params", "ChemoReceptors.lambda") : NormalDistTuple(mu=-54000, sigma=3000.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : NormalDistTuple(mu=-2.10, sigma=1.0),
	ModParam("params", "Head.neurons.AIY.theta") : NormalDistTuple(mu=-2.76, sigma=1.0),
	ModParam("params", "Head.neurons.RIA.theta") : NormalDistTuple(mu=-2.46, sigma=1.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}


dists_chemo_v15_1 : ModParamsDists = {
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : NormalDistTuple(mu=230.0, sigma=15.0),
	ModParam("params", "ChemoReceptors.lambda") : NormalDistTuple(mu=-52000, sigma=3000.0),
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : NormalDistTuple(mu=-10.0, sigma=3.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : NormalDistTuple(mu=-1.0, sigma=1.0),
	ModParam("conn",   "Head,AIY,RIA,chem") : NormalDistTuple(mu=-40.0, sigma=15.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : NormalDistTuple(mu=-15.5, sigma=2.0),
	ModParam("conn",   "Head,SMD*,RIA,chem") : NormalDistTuple(mu=1.3, sigma=1.0),
	ModParam("conn",   "Head,RIA,RIA,chem") : NormalDistTuple(mu=0.0, sigma=0.1),
	ModParam("conn",   "Head,RIA,SMD*,chem") : NormalDistTuple(mu=0.0, sigma=20.0),
	ModParam("conn",   "Head,RMD*,RIA,chem") : NormalDistTuple(mu=0.0, sigma=1.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : NormalDistTuple(mu=-2.10, sigma=1.0),
	ModParam("params", "Head.neurons.AIY.theta") : NormalDistTuple(mu=-2.76, sigma=1.0),
	ModParam("params", "Head.neurons.RIA.theta") : NormalDistTuple(mu=-2.46, sigma=1.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}

dists_chemo_v16_1 : ModParamsDists = {
	# chemo params
	ModParam("params", "ChemoReceptors.kappa") : NormalDistTuple(mu=230.0, sigma=15.0),
	ModParam("params", "ChemoReceptors.lambda") : NormalDistTuple(mu=-52000, sigma=3000.0),
	# conns
	ModParam("conn",   "Head,AWA,AIY,chem") : NormalDistTuple(mu=-10.0, sigma=3.0),
	ModParam("conn",   "Head,AIY,AIY,chem") : NormalDistTuple(mu=-1.0, sigma=1.0),
	ModParam("conn",   "Head,AIY,RIA,chem") : NormalDistTuple(mu=-40.0, sigma=15.0),
	ModParam("conn",   "Head,RIA,RMD*,chem") : NormalDistTuple(mu=-15.5, sigma=2.0),
	ModParam("conn",   "Head,SMD*,RIA,chem") : NormalDistTuple(mu=1.5, sigma=1.0),
	ModParam("conn",   "Head,RIA,RIA,chem") : NormalDistTuple(mu=0.0, sigma=0.1),
	ModParam("conn",   "Head,RIA,SMD*,chem") : NormalDistTuple(mu=10.0, sigma=20.0),
	ModParam("conn",   "Head,RMD*,RIA,chem") : NormalDistTuple(mu=0.0, sigma=1.0),
	# neuron params
	ModParam("params", "Head.neurons.AWA.theta") : NormalDistTuple(mu=-2.0, sigma=1.0),
	ModParam("params", "Head.neurons.AIY.theta") : NormalDistTuple(mu=-1.0, sigma=1.0),
	ModParam("params", "Head.neurons.RIA.theta") : NormalDistTuple(mu=-2.0, sigma=1.0),
	# ModParam("params", "") : RangeTuple(,),
	# ModParam("conn",   "") : RangeTuple(,),
}


# DEFAULT_RANGES : ModParamsRanges = ranges_chemo_v12_1
DEFAULT_DISTS : ModParamsDists = dists_chemo_v15_1







DEFAULT_EVALRUNS : List[ModParamsDict] = [
	{ ModParam("params",   "simulation.angle") : 0.5 },
	{ ModParam("params",   "simulation.angle") : -0.5 },
	{ ModParam("params",   "simulation.angle") : 1.0 },
	{ ModParam("params",   "simulation.angle") : -1.0 },
	{ ModParam("params",   "simulation.angle") : 0.2 },
	{ ModParam("params",   "simulation.angle") : -0.2 },
]