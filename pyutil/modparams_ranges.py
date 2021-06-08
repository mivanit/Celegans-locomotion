from typing import *

if TYPE_CHECKING:
	from pyutil.util import (
		ModParam, ModTypes, Path,mkdir,joinPath,
		strList_to_dict,ParamsDict,ModParamsDict,ModParamsRanges,
		RangeTuple,
	)
else:
	from util import (
		ModParam, ModTypes, Path,mkdir,joinPath,
		strList_to_dict,ParamsDict,ModParamsDict,ModParamsRanges,
		RangeTuple,
	)

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


DEFAULT_RANGES : ModParamsRanges = ranges_chemo_v7_1