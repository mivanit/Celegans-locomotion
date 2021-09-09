---
header-includes: "<style>\nbody {\n  max-width: 50em;\n}\n</style>"
title: todo-inline

metadata:
  files_with_todos: 19
  num_items: 86
  num_unique_tags: 7
  searched_files: 43

# suggested command for conversion to html
cmd: "pandoc todo-inline.md -o todo-inline.html --from markdown+backtick_code_blocks+fenced_code_attributes --standalone --toc --toc-depth 1"
---
# **BUG** -- 20 items
## [`../modules/Collide.cpp`](../modules/Collide.cpp) -- 1 item
 - [ ] BUG("    >> elements in CollObjs vec: %ld\n", CollObjs.size()) 
	(line 115)
	
	<details>
	```{.cpp .numberLines startFrom="115"}
	PRINTF_DEBUG("    >> elements in CollObjs vec: %ld\n", CollObjs.size())
	for (CollisionObject obj : CollObjs)
	{
	    if (obj.coll_type == Box_Ax)
	```
	</details>

## [`../modules/NervousSystem.cpp`](../modules/NervousSystem.cpp) -- 4 items
 - [ ] BUG("    > circuit init\n") 
	(line 30)
	
	<details>
	```{.cpp .numberLines startFrom="30"}
	PRINT_DEBUG("    > circuit init\n")
	// compute and set the circuit size
	SetCircuitSize(
	    compute_size(ns_data["neurons"]),
	    compute_maxconn_bidir_sum(ns_data["connections"], CONNTYPE_CHEM) + 1,
	```
	</details>

 - [ ] BUG("    > loading neurons\n") 
	(line 40)
	
	<details>
	```{.cpp .numberLines startFrom="40"}
	PRINT_DEBUG("    > loading neurons\n")
	namesMapInv = std::vector<string>(size + 1, "NULL");
	loadJSON_neurons(ns_data["neurons"]);
	// load the connections
	```
	</details>

 - [ ] BUG("    > adding synapses\n") 
	(line 45)
	
	<details>
	```{.cpp .numberLines startFrom="45"}
	PRINT_DEBUG("    > adding synapses\n")
	for (auto syn : ns_data["connections"])
	{
	    AddSynapse_JSON(syn);
	}
	```
	</details>

 - [ ] BUG("    > circuit init\n") 
	(line 55)
	
	<details>
	```{.cpp .numberLines startFrom="55"}
	PRINT_DEBUG("    > circuit init\n")
	
	// compute and set the circuit size
	int unit_size = compute_size(ns_data["neurons"]);
	// REVIEW: connection count calculation is bugged :/
	```
	</details>

## [`../modules/Worm.cpp`](../modules/Worm.cpp) -- 10 items
 - [ ] BUG("  > muscles\n") 
	(line 178)
	
	<details>
	```{.cpp .numberLines startFrom="178"}
	PRINT_DEBUG("  > muscles\n")
	m.SetMuscleParams(N_muscles, T_muscle);
	// PRINT_DEBUG("  > validating NS params\n")
	// params["Head"]
	```
	</details>

 - [ ] BUG("  > Head NS\n") 
	(line 186)
	
	<details>
	```{.cpp .numberLines startFrom="186"}
	PRINT_DEBUG("  > Head NS\n")
	h.init_NS(params["Head"]);
	// VC Circuit
	PRINT_DEBUG("  > VentralCord NS\n")
	n.init_NS_repeatedUnits(params["VentralCord"], N_units);
	```
	</details>

 - [ ] BUG("  > VentralCord NS\n") 
	(line 189)
	
	<details>
	```{.cpp .numberLines startFrom="189"}
	PRINT_DEBUG("  > VentralCord NS\n")
	n.init_NS_repeatedUnits(params["VentralCord"], N_units);
	PRINT_DEBUG("  > updating neuron indecies\n")
	```
	</details>

 - [ ] BUG("  > updating neuron indecies\n") 
	(line 193)
	
	<details>
	```{.cpp .numberLines startFrom="193"}
	PRINT_DEBUG("  > updating neuron indecies\n")
	SMDD = h.namesMap["SMDD"];
	RMDD = h.namesMap["RMDD"];
	SMDV = h.namesMap["SMDV"];
	RMDV = h.namesMap["RMDV"];
	```
	</details>

 - [ ] BUG("  > Stretch Receptors\n") 
	(line 207)
	
	<details>
	```{.cpp .numberLines startFrom="207"}
	PRINT_DEBUG("  > Stretch Receptors\n")
	sr.SetStretchReceptorParams(
	    N_segments, 
	    N_stretchrec, 
	    params["StretchReceptors"]["VC_gain"].get<double>(), 
	```
	</details>

 - [ ] BUG("  > NMJ params\n") 
	(line 253)
	
	<details>
	```{.cpp .numberLines startFrom="253"}
	PRINT_DEBUG("  > NMJ params\n")
	NMJ_DB = params["NMJ"]["DB"];
	NMJ_VBa = params["NMJ"]["VBA"];
	NMJ_VBp = params["NMJ"]["VBP"];
	NMJ_DD = params["NMJ"]["DD"];
	```
	</details>

 - [ ] BUG("  > NMJ gain (?)\n") 
	(line 268)
	
	<details>
	```{.cpp .numberLines startFrom="268"}
	PRINT_DEBUG("  > NMJ gain (?)\n")
	NMJ_Gain_Map = 0.5;
	NMJ_Gain.SetBounds(1, N_muscles);
	for (int i=1; i<=N_muscles; i++)
	{
	```
	</details>

 - [ ] BUG("  > Worm object ctor done!\n") 
	(line 275)
	
	<details>
	```{.cpp .numberLines startFrom="275"}
	    PRINT_DEBUG("  > Worm object ctor done!\n")
	}
	```
	</details>

 - [ ] BUG("    > circuit state\n") 
	(line 283)
	
	<details>
	```{.cpp .numberLines startFrom="283"}
	PRINT_DEBUG("    > circuit state\n")
	t = 0.0;
	n.RandomizeCircuitState(-0.5, 0.5, rs);
	h.RandomizeCircuitState(-0.5, 0.5, rs);
	PRINTF_DEBUG("    > body state\n      >> angle: %f, collision obj count: %ld\n", angle, collObjs.size())
	```
	</details>

 - [ ] BUG("    > muscle state\n") 
	(line 289)
	
	<details>
	```{.cpp .numberLines startFrom="289"}
	    PRINT_DEBUG("    > muscle state\n")
	    m.InitializeMuscleState();
	}
	void Worm::HeadStep(double StepSize, double output)
	```
	</details>

## [`../modules/WormBody.cpp`](../modules/WormBody.cpp) -- 1 item
 - [ ] BUG 
	(line 38)
	
	<details>
	```{.cpp .numberLines startFrom="38"}
	#ifdef DEBUG
	    #include <iostream>
	#endif
	```
	</details>

## [`../modules/VectorMatrix.h`](../modules/VectorMatrix.h) -- 3 items
 - [ ] BUG 0 
	(line 16)
	
	<details>
	```{.c .numberLines startFrom="16"}
	#define DEBUG 0
	// *******
	// TVector
	// *******
	```
	</details>

 - [ ] BUG 
	(line 290)
	
	<details>
	```{.c .numberLines startFrom="290"}
	#if !DEBUG
	        return Matrix[index];
	#else
	        if (index < lb1 || index > ub1)
	        {
	```
	</details>

 - [ ] BUG 
	(line 51)
	
	<details>
	```{.c .numberLines startFrom="51"}
	#if !DEBUG
	        return Vector[index];
	#else
	        return (*this)(index);
	#endif
	```
	</details>

## [`../pyutil/diffusion/Collide_standalone.h`](../pyutil/diffusion/Collide_standalone.h) -- 1 item
 - [ ] BUG("    >> elements in CollObjs vec: %ld\n", CollObjs.size()) 
	(line 224)
	
	<details>
	```{.c .numberLines startFrom="224"}
	PRINTF_DEBUG("    >> elements in CollObjs vec: %ld\n", CollObjs.size())
	for (CollisionObject obj : CollObjs)
	{
	    if (obj.coll_type == Box_Ax)
	```
	</details>

# **REVIEW** -- 15 items
## [`../modules/NervousSystem.cpp`](../modules/NervousSystem.cpp) -- 2 items
 - [ ] REVIEW: why... why is this 1-indexed 
	(line 128)
	
	<details>
	```{.cpp .numberLines startFrom="128"}
	// REVIEW: why... why is this 1-indexed
	states.SetBounds(1,size);
	states.FillContents(0.0);
	paststates.SetBounds(1,size);  
	paststates.FillContents(0.0);
	```
	</details>

 - [ ] REVIEW: connection count calculation is bugged :/ 
	(line 59)
	
	<details>
	```{.cpp .numberLines startFrom="59"}
	// REVIEW: connection count calculation is bugged :/
	// TODO: instead of adding the maxs together, merge the lists and take the max once
	int max_CHEM = compute_maxconn_bidir(ns_data["connections"], CONNTYPE_CHEM) 
	    + compute_maxconn_bidir_sum(ns_data["connections_fwd"], CONNTYPE_CHEM) + 1;
	int max_ELE = compute_maxconn_bidir_sum(ns_data["connections"], CONNTYPE_ELE) 
	```
	</details>

## [`../modules/Worm.cpp`](../modules/Worm.cpp) -- 2 items
 - [ ] REVIEW: not very clean 
	(line 252)
	
	<details>
	```{.cpp .numberLines startFrom="252"}
	// REVIEW: not very clean 
	PRINT_DEBUG("  > NMJ params\n")
	NMJ_DB = params["NMJ"]["DB"];
	NMJ_VBa = params["NMJ"]["VBA"];
	NMJ_VBp = params["NMJ"]["VBP"];
	```
	</details>

 - [ ] REVIEW: what is this doing? 
	(line 267)
	
	<details>
	```{.cpp .numberLines startFrom="267"}
	// REVIEW: what is this doing?
	PRINT_DEBUG("  > NMJ gain (?)\n")
	NMJ_Gain_Map = 0.5;
	NMJ_Gain.SetBounds(1, N_muscles);
	for (int i=1; i<=N_muscles; i++)
	```
	</details>

## [`../modules/WormBody.cpp`](../modules/WormBody.cpp) -- 1 item
 - [ ] REVIEW: this parameter can be tuned 
	(line 57)
	
	<details>
	```{.cpp .numberLines startFrom="57"}
	// REVIEW: this parameter can be tuned
	double radius_check;
	// Initialize various vectors of per rod body constants
	```
	</details>

## [`../modules/Collide.h`](../modules/Collide.h) -- 2 items
 - [ ] REVIEW: unclear what this does 
	(line 169)
	
	<details>
	```{.c .numberLines startFrom="169"}
	// REVIEW: unclear what this does
	  realtype P_x,P_y,Distance,magF,D_scale,magF_P1,magF_P2;
	// reset contact force
	  ContactForce = 0;
	```
	</details>

 - [ ] REVIEW: this would be cleaner if I used polymorphism properly lol 
	(line 31)
	
	<details>
	```{.c .numberLines startFrom="31"}
	// REVIEW: this would be cleaner if I used polymorphism properly lol
	struct CollisionObject
	{
	    CollisionType coll_type;
	```
	</details>

## [`../modules/NervousSystem.h`](../modules/NervousSystem.h) -- 1 item
 - [ ] REVIEW: for some reason these cause a "multiple definitions" error unless they are inlined 
	(line 56)
	
	<details>
	```{.c .numberLines startFrom="56"}
	// REVIEW: for some reason these cause a "multiple definitions" error unless they are inlined
	inline int compute_size(json & neurons)
	{
	    return std::distance(neurons.begin(), neurons.end());
	}
	```
	</details>

## [`../pyutil/diffusion/Collide_standalone.h`](../pyutil/diffusion/Collide_standalone.h) -- 1 item
 - [ ] REVIEW: this would be cleaner if I used polymorphism properly lol 
	(line 19)
	
	<details>
	```{.c .numberLines startFrom="19"}
	// REVIEW: this would be cleaner if I used polymorphism properly lol
	struct CollisionObject
	{
	    CollisionType coll_type;
	```
	</details>

## [`../pyutil/eval_run.py`](../pyutil/eval_run.py) -- 1 item
 - [ ] REVIEW: very fragile 
	(line 267)
	
	<details>
	```{.python .numberLines startFrom="267"}
	# REVIEW: very fragile
	# TODO: make less fragile
	# UGLY: very fragile
	```
	</details>

## [`../pyutil/genetic_utils.py`](../pyutil/genetic_utils.py) -- 2 items
 - [ ] REVIEW: I think this actually makes the mutations too small 
	(line 658)
	
	<details>
	```{.python .numberLines startFrom="658"}
	# REVIEW: I think this actually makes the mutations too small
	
	ranges : ModParamsRanges = dict()
	
	if ranges_override is None:
	```
	</details>

 - [ ] REVIEW: this return 
	(line 935)
	
	<details>
	```{.python .numberLines startFrom="935"}
	# REVIEW: this return
	# return pop[0]
	```
	</details>

## [`../pyutil/params.py`](../pyutil/params.py) -- 3 items
 - [ ] REVIEW: this isnt full regex, but whatever 
	(line 207)
	
	<details>
	```{.python .numberLines startFrom="207"}
	# REVIEW: this isnt full regex, but whatever
	if nrn.startswith(conn_key['to'].split('*')[0]):
	    conn_key_temp['to'] = nrn
	    cidx_temp = find_conn_idx(
	        params_data[conn_key_temp['NS']]['connections'],
	```
	</details>

 - [ ] REVIEW: this isnt full regex, but whatever 
	(line 224)
	
	<details>
	```{.python .numberLines startFrom="224"}
	# REVIEW: this isnt full regex, but whatever
	if nrn.startswith(conn_key['from'].split('*')[0]):
	    conn_key_temp['from'] = nrn
	    cidx_temp = find_conn_idx(
	        params_data[conn_key_temp['NS']]['connections'],
	```
	</details>

 - [ ] REVIEW: why did i even refactor this when im making everything editable through params json anyway? 
	(line 299)
	
	<details>
	```{.python .numberLines startFrom="299"}
	# REVIEW: why did i even refactor this when im making everything editable through params json anyway?
	for tup_key,val in params_mod.items():
	    # merge in the standard params
	    if tup_key.mod_type == ModTypes.params.value:
	        
	```
	</details>

# **TODO** -- 30 items
## [`../modules/NervousSystem.cpp`](../modules/NervousSystem.cpp) -- 2 items
 - [ ] TODO: when switching to 0-idx, remove the +1 here 
	(line 393)
	
	<details>
	```{.cpp .numberLines startFrom="393"}
	// TODO: when switching to 0-idx, remove the +1 here
	int idx = idx_shift + i + 1;
	// if the shift index is nonzero, dont add the names to the map
	PRINTF_DEBUG_NRNDET("        >>  creating neuron %s at idx %d\n", nrn.key().c_str(), idx)
	if (idx_shift == 0)
	```
	</details>

 - [ ] TODO: instead of adding the maxs together, merge the lists and take the max once 
	(line 60)
	
	<details>
	```{.cpp .numberLines startFrom="60"}
	// TODO: instead of adding the maxs together, merge the lists and take the max once
	int max_CHEM = compute_maxconn_bidir(ns_data["connections"], CONNTYPE_CHEM) 
	    + compute_maxconn_bidir_sum(ns_data["connections_fwd"], CONNTYPE_CHEM) + 1;
	int max_ELE = compute_maxconn_bidir_sum(ns_data["connections"], CONNTYPE_ELE) 
	    + compute_maxconn_bidir_sum(ns_data["connections_fwd"], CONNTYPE_ELE);
	```
	</details>

## [`../modules/Worm.cpp`](../modules/Worm.cpp) -- 1 item
 - [ ] TODO: dynamically read from hash map 
	(line 354)
	
	<details>
	```{.cpp .numberLines startFrom="354"}
	// TODO: dynamically read from hash map
	// Set input to Muscles
	//  Input from the head circuit
	dorsalHeadInput = NMJ_SMDD*h.NeuronOutput(SMDD) + NMJ_RMDV*h.NeuronOutput(RMDD);
	ventralHeadInput = NMJ_SMDV*h.NeuronOutput(SMDV) + NMJ_RMDD*h.NeuronOutput(RMDV);
	```
	</details>

## [`../modules/Collide.h`](../modules/Collide.h) -- 3 items
 - [ ] TODO: do_collide_friction function 
	(line 136)
	
	<details>
	```{.c .numberLines startFrom="136"}
	// TODO: do_collide_friction function
	// loop over all the objects and all the points
	// and check for collisions
	inline std::vector<VecXY> do_collide_vec(std::vector<VecXY> & pos_vec, std::vector<CollisionObject> & objs_vec);
	```
	</details>

 - [ ] TODO: instead of checking distance, check inside collision box (see existing python code) 
	(line 199)
	
	<details>
	```{.c .numberLines startFrom="199"}
	// TODO: instead of checking distance, check inside collision box (see existing python code)
	if(Distance < Objects[k][2]){
	    magF = (
	        k_Object * (Objects[k][2] - Distance) 
	        + D_scale*k_Object * (pow((Objects[k][2] - Distance) / D_scale,2))
	```
	</details>

 - [ ] TODO: this needs to call the data structure used in the Izquierdo code 
	(line 225)
	
	<details>
	```{.c .numberLines startFrom="225"}
	// TODO: this needs to call the data structure used in the Izquierdo code
	  for(int i = 1; i < NSEG; ++i){
	    int i_minus_1 = i-1;
	    F_term[i][0][0] = F_H[i_minus_1][0]*Dir[i_minus_1][0][0] - F_H[i][0]*Dir[i][0][0] + F_D[i_minus_1][1]*Dir_D[i_minus_1][1][0] - F_D[i][0]*Dir_D[i][0][0] + F_object[i][0][0];
	```
	</details>

## [`../modules/NervousSystem.h`](../modules/NervousSystem.h) -- 1 item
 - [ ] TODO: deprecate 
	(line 161)
	
	<details>
	```{.c .numberLines startFrom="161"}
	// TODO: deprecate
	std::unordered_map<string,int> namesMap;
	std::vector<string> namesMapInv;
	
	void AddSynapse_JSON(json & syn, int idx_shift_A = 0, int idx_shift_B = 0);
	```
	</details>

## [`../modules/StretchReceptor.h`](../modules/StretchReceptor.h) -- 1 item
 - [ ] TODO: make this more accurate -- corner distance, or full diffusion sim 
	(line 95)
	
	<details>
	```{.c .numberLines startFrom="95"}
	// TODO: make this more accurate -- corner distance, or full diffusion sim
	// set the concentration to zero if it is more than some max distance away
	// OPTIMIZE: make branchless by multiplying gradient by bool
	double food_dist_sqrd = dist_sqrd(headpos, foodpos);
	if (pow(dist_sqrd(headpos, foodpos), 0.5) > max_distance)
	```
	</details>

## [`../modules/packages/cxxopts.hpp`](../modules/packages/cxxopts.hpp) -- 2 items
 - [ ] TODO: maybe default options should count towards the number of arguments 
	(line 1163)
	
	<details>
	```{.cpp .numberLines startFrom="1163"}
	// TODO: maybe default options should count towards the number of arguments
	CXXOPTS_NODISCARD
	bool
	has_default() const noexcept
	{
	```
	</details>

 - [ ] TODO: remove the duplicate code here 
	(line 1852)
	
	<details>
	```{.cpp .numberLines startFrom="1852"}
	  // TODO: remove the duplicate code here
	  auto& store = m_parsed[details->hash()];
	  store.parse_default(details);
	}
	```
	</details>

## [`../pyutil/diffusion/Collide_standalone.h`](../pyutil/diffusion/Collide_standalone.h) -- 1 item
 - [ ] TODO: do_collide_friction function 
	(line 323)
	
	<details>
	```{.c .numberLines startFrom="323"}
	// TODO: do_collide_friction function
	```
	</details>

## [`../pyutil/collision_object.py`](../pyutil/collision_object.py) -- 1 item
 - [ ] TODO (minor): type hint to `CollisionType` doesn't have expected behavior, this is just because python enum type hints are weird 
	(line 48)
	
	<details>
	```{.python .numberLines startFrom="48"}
	# TODO (minor): type hint to `CollisionType` doesn't have expected behavior, this is just because python enum type hints are weird
	class CollisionObject(object):
	    
	    ATTRIBUTES : Dict[CollisionType,List[str]] = {
	```
	</details>

## [`../pyutil/eval_run.py`](../pyutil/eval_run.py) -- 7 items
 - [ ] TODO: ret_nan is not actually doing the correct thing here, although its probably unimportant. current implementation does not allow for just one of several processes failing 
	(line 114)
	
	<details>
	```{.python .numberLines startFrom="114"}
	# TODO: ret_nan is not actually doing the correct thing here, although its probably unimportant. current implementation does not allow for just one of several processes failing
	extracted : Dict[Path,float] = dict()
	for p in os.listdir(datadir):
	```
	</details>

 - [ ] TODO: implement extracting more data, for parameter sweeps 
	(line 258)
	
	<details>
	```{.python .numberLines startFrom="258"}
	    # TODO: implement extracting more data, for parameter sweeps
	    raise NotImplementedError('please implement me :(')
	def calcmean_symmetric(data : Dict[str,float]) -> float:
	```
	</details>

 - [ ] TODO: make less fragile 
	(line 268)
	
	<details>
	```{.python .numberLines startFrom="268"}
	# TODO: make less fragile
	# UGLY: very fragile
	# get the angles and match with pairs
	```
	</details>

 - [ ] TODO: use params json instead? 
	(line 278)
	
	<details>
	```{.python .numberLines startFrom="278"}
	# TODO: use params json instead?
	k_dict : Dict[str,str] = dict_from_dirname(k, func_cast = str)
	# take absolute value
	angle : str = k_dict['angle'].strip('- ')
	```
	</details>

 - [ ] TODO: assert 2 elements in each list 
	(line 286)
	
	<details>
	```{.python .numberLines startFrom="286"}
	# TODO: assert 2 elements in each list
	# TODO: assert all angles present
	# TODO: handle angle keys better?
	# min of each pair, then average
	```
	</details>

 - [ ] TODO: assert all angles present 
	(line 287)
	
	<details>
	```{.python .numberLines startFrom="287"}
	# TODO: assert all angles present
	# TODO: handle angle keys better?
	# min of each pair, then average
	per_angle_min : Dict[str,float] = {
	```
	</details>

 - [ ] TODO: handle angle keys better? 
	(line 288)
	
	<details>
	```{.python .numberLines startFrom="288"}
	# TODO: handle angle keys better?
	# min of each pair, then average
	per_angle_min : Dict[str,float] = {
	    k : min(lst_v)
	```
	</details>

## [`../pyutil/genetic_utils.py`](../pyutil/genetic_utils.py) -- 8 items
 - [ ] TODO: review this function type 
	(line 177)
	
	<details>
	```{.python .numberLines startFrom="177"}
	# TODO: review this function type
	GenoCombineFunc = Callable
	# GenoCombineFunc = Callable[
	#     [ModParamsDict, ModParamsDict],
	#     ModParamsDict,
	```
	</details>

 - [ ] TODO: for some reason, `popsize_old` sometimes is less than or equal to zero. the max() is just a hack, since i dont know what causes the issue in the first place 
	(line 291)
	
	<details>
	```{.python .numberLines startFrom="291"}
	# TODO: for some reason, `popsize_old` sometimes is less than or equal to zero. the max() is just a hack, since i dont know what causes the issue in the first place
	# random_selection : NDArray = np.random.randint(
	#     low = 0, 
	#     high = max(1,popsize_old), 
	#     size = (popsize_new, 2),
	```
	</details>

 - [ ] TODO: not the real median here, oops 
	(line 523)
	
	<details>
	```{.python .numberLines startFrom="523"}
	# TODO: not the real median here, oops
	if lst_fit:
	    return {
	        'max' : lst_fit[0],
	        'median' : lst_fit[len(lst_fit) // 2],
	```
	</details>

 - [ ] TODO: fix typing here 
	(line 559)
	
	<details>
	```{.python .numberLines startFrom="559"}
	# TODO: fix typing here
	assert not any(
	    f is None
	    for _,f in pop
	), "`None` fitness found when trying to run `generation_selection`"
	```
	</details>

 - [ ] TODO: pop/push if the element count is not quite right? 
	(line 587)
	
	<details>
	```{.python .numberLines startFrom="587"}
	# TODO: pop/push if the element count is not quite right?
	return newpop
	
	```
	</details>

 - [ ] TODO: document this 
	(line 61)
	
	<details>
	```{.python .numberLines startFrom="61"}
	# TODO: document this
	
	# make dir
	if out_name is None:
	    outpath : Path = joinPath(rootdir, dict_hash(params_mod))
	```
	</details>

 - [ ] TODO: check that params jsons match. this also means that strip-only keys dont mean anything at the moment 
	(line 768)
	
	<details>
	```{.python .numberLines startFrom="768"}
	# TODO: check that params jsons match. this also means that strip-only keys dont mean anything at the moment
	if params_ref is not None:
	    raise NotImplementedError('checking that params match is not yet implemented')
	# for p in lst_params_stripped:
	#     for k,v in p.items():
	```
	</details>

 - [ ] TODO: `pop_sizes` should be an input parameter 
	(line 882)
	
	<details>
	```{.python .numberLines startFrom="882"}
	# TODO: `pop_sizes` should be an input parameter
	pop_sizes : List[Tuple[int, int]] = compute_gen_sizes(
	    first_gen_size = first_gen_size,
	    gen_count = gen_count,
	    factor_cull = factor_cull,
	```
	</details>

## [`../pyutil/read_runs.py`](../pyutil/read_runs.py) -- 2 items
 - [ ] TODO: load eval runs params 
	(line 349)
	
	<details>
	```{.python .numberLines startFrom="349"}
	# TODO: load eval runs params
	# TODO: store intersection of params, collobjs
	output['rootdir'] = unixPath(rootdir)
	```
	</details>

 - [ ] TODO: store intersection of params, collobjs 
	(line 351)
	
	<details>
	```{.python .numberLines startFrom="351"}
	# TODO: store intersection of params, collobjs
	output['rootdir'] = unixPath(rootdir)
	return output
	```
	</details>

## [`../pyutil/plot/pos.py`](../pyutil/plot/pos.py) -- 1 item
 - [ ] TODO: make this actually reference matplotlib.Axes 
	(line 55)
	
	<details>
	```{.python .numberLines startFrom="55"}
	# TODO: make this actually reference matplotlib.Axes
	Axes = Any
	OptInt = Optional[int]
	```
	</details>

# **NOTE** -- 9 items
## [`../modules/WormBody.cpp`](../modules/WormBody.cpp) -- 2 items
 - [ ] NOTE 2: There is a discrepency between the BBC code and paper as to whether or not the factor of 2 here 
	(line 13)
	
	<details>
	```{.cpp .numberLines startFrom="13"}
	//   NOTE 2: There is a discrepency between the BBC code and paper as to whether or not the factor of 2 here
	//   should be raised to the fourth power (16). I have followed the BBC code in raising it to the fourth power
	//   (Boyle, personal communication, June 30, 2014).
	//
	//   NOTE 3: There is a major discrepency between the BBC code on the one hand and the BBC paper and Boyle thesis
	```
	</details>

 - [ ] NOTE 3: There is a major discrepency between the BBC code on the one hand and the BBC paper and Boyle thesis 
	(line 17)
	
	<details>
	```{.cpp .numberLines startFrom="17"}
	//   NOTE 3: There is a major discrepency between the BBC code on the one hand and the BBC paper and Boyle thesis
	//   on the other as to the formula for dphi/dt. The BBC code uses (f_odd_par/C_par)/(M_PI*2.0*R[i]), but both documents
	//   use 2*f_odd_par/(R[i]*C_par). The argument for the latter formula makes perfect sense to me (see Boyle Thesis,
	//   p. 76, 128). However, I cannot understand where the first forumula comes from and Boyle was unable to shed any
	//   light on the matter (Boyle, personal communication, June 30, 2014). Which formula used does make a small but nontrivial
	```
	</details>

## [`../modules/Collide.h`](../modules/Collide.h) -- 2 items
 - [ ] NOTE: this printf statement is cursed. somehow `(fabs(y) > EPSILON) ? "true" : "false"` evaluated to "sin" before causing a segfault. no clue what was going on. 
	(line 87)
	
	<details>
	```{.c .numberLines startFrom="87"}
	// NOTE: this printf statement is cursed. somehow `(fabs(y) > EPSILON) ? "true" : "false"` evaluated to "sin" before causing a segfault. no clue what was going on.
	// printf(
	//     "is_nonzero:\t%f,%f,%f\t%s,%s\n", 
	//     fabs(x), fabs(y), EPSILON,
	//     (fabs(x) > EPSILON) ? "true" : "false", 
	```
	</details>

 - [ ] NOTE: for some reason, abs() doesnt work and casts things to ints 
	(line 94)
	
	<details>
	```{.c .numberLines startFrom="94"}
	    // NOTE: for some reason, abs() doesnt work and casts things to ints
	    // std::cout << std::fixed;
	    // std::cout << std::setprecision(5) << "is_nonzero:\t" << fabs(x) << "," << fabs(y) << ","  << EPSILON << ","  << ((fabs(x) > EPSILON) ? "true" : "false") << ","  << ((fabs(y) > EPSILON) ? "true" : "false") << std::endl;
	}
	```
	</details>

## [`../modules/WormBody.h`](../modules/WormBody.h) -- 1 item
 - [ ] NOTE 1: I fixed an obvious C macro bug in the original BBC code. In that code, NBAR is defined to be "NSEG+1", 
	(line 13)
	
	<details>
	```{.c .numberLines startFrom="13"}
	//   NOTE 1: I fixed an obvious C macro bug in the original BBC code. In that code, NBAR is defined to be "NSEG+1",
	//   but then used in, e.g., "2.0*NBAR", which expands into "2.0*NSEG+1" rather than "2.0*(NSEG+1)".
	//
	// Created by Randall Beer on 7/8/14.
	// Copyright (c) 2014 Randall Beer. All rights reserved.
	```
	</details>

## [`../pyutil/diffusion/Collide_standalone.h`](../pyutil/diffusion/Collide_standalone.h) -- 2 items
 - [ ] NOTE: this printf statement is cursed. somehow `(fabs(y) > EPSILON) ? "true" : "false"` evaluated to "sin" before causing a segfault. no clue what was going on. 
	(line 75)
	
	<details>
	```{.c .numberLines startFrom="75"}
	// NOTE: this printf statement is cursed. somehow `(fabs(y) > EPSILON) ? "true" : "false"` evaluated to "sin" before causing a segfault. no clue what was going on.
	// printf(
	//     "is_nonzero:\t%f,%f,%f\t%s,%s\n", 
	//     fabs(x), fabs(y), EPSILON,
	//     (fabs(x) > EPSILON) ? "true" : "false", 
	```
	</details>

 - [ ] NOTE: for some reason, abs() doesnt work and casts things to ints 
	(line 82)
	
	<details>
	```{.c .numberLines startFrom="82"}
	    // NOTE: for some reason, abs() doesnt work and casts things to ints
	    // std::cout << std::fixed;
	    // std::cout << std::setprecision(5) << "is_nonzero:\t" << fabs(x) << "," << fabs(y) << ","  << EPSILON << ","  << ((fabs(x) > EPSILON) ? "true" : "false") << ","  << ((fabs(y) > EPSILON) ? "true" : "false") << std::endl;
	}
	```
	</details>

## [`../pyutil/old/WormView.m`](../pyutil/old/WormView.m) -- 2 items
 - [ ] NOTE: Create figure then check box to fill axes! 
	(line 38)
	
	<details>
	```{.c .numberLines startFrom="38"}
	%NOTE: Create figure then check box to fill axes!
	figure('Position',[1 1 824 588])
	XYratio = 1.3333;   %This is the ratio of Xrange/Yrange required to give equal axes
	R = D/2.0*abs(sin(acos(((0:Nbar-1)-NSEG./2.0)./(NSEG/2.0 + 0.2))));
	```
	</details>

 - [ ] NOTE: if not using objects, you must delete any objects.csv file generated 
	(line 8)
	
	<details>
	```{.c .numberLines startFrom="8"}
	%NOTE: if not using objects, you must delete any objects.csv file generated
	%during previous runs.
	usingObjects = exist('objects.csv','file');
	if usingObjects 
	    Objects = importdata('objects.csv');
	```
	</details>

# **OPTIMIZE** -- 5 items
## [`../modules/WormBody.cpp`](../modules/WormBody.cpp) -- 1 item
 - [ ] OPTIMIZE: can do clever chunking by checking only a subset of points against each object 
	(line 285)
	
	<details>
	```{.cpp .numberLines startFrom="285"}
	// OPTIMIZE: can do clever chunking by checking only a subset of points against each object
	// loop over all collision boxes
	for ( CollisionObject obj : CollObjs )
	{
	    
	```
	</details>

## [`../pyutil/genetic_utils.py`](../pyutil/genetic_utils.py) -- 2 items
 - [ ] OPTIMIZE: this uses two loops (because its just a view over the list) when it could be using just one 
	(line 149)
	
	<details>
	```{.python .numberLines startFrom="149"}
	# OPTIMIZE: this uses two loops (because its just a view over the list) when it could be using just one
	output : ModParamsRanges = {
	    key : RangeTuple(
	        min(p[key] for p in pop),
	        max(p[key] for p in pop),
	```
	</details>

 - [ ] OPTIMIZE: each process could be read as it finishes -- this is not perfectly efficient 
	(line 485)
	
	<details>
	```{.python .numberLines startFrom="485"}
	# OPTIMIZE: each process could be read as it finishes -- this is not perfectly efficient
	for p in lst_proc:
	    p.wait()
	new_fit : float = func_extract_multi(
	```
	</details>

## [`../pyutil/plot/gene.py`](../pyutil/plot/gene.py) -- 1 item
 - [ ] OPTIMIZE: only scrape the ones with gen > min_gen 
	(line 210)
	
	<details>
	```{.python .numberLines startFrom="210"}
	# OPTIMIZE: only scrape the ones with gen > min_gen
	data : Dict[GeneRunID, float] = scrape_extracted_cache(rootdir)
	# first make bins based on all data
	bins,bin_centers = get_bins(data, n_bins)
	```
	</details>

## [`../pyutil/plot/pos.py`](../pyutil/plot/pos.py) -- 1 item
 - [ ] OPTIMIZE: this bit can be vectorized 
	(line 208)
	
	<details>
	```{.python .numberLines startFrom="208"}
	# OPTIMIZE: this bit can be vectorized
	for t in range(n_tstep):
	    dX : float = worm_thickness * np.cos(data[t]['phi'])
	    dY : float = worm_thickness * np.sin(data[t]['phi'])
	    data_Dorsal[t]['x'] = data[t]['x'] + dX
	```
	</details>

# **HACK** -- 1 item
## [`../pyutil/collision_object.py`](../pyutil/collision_object.py) -- 1 item
 - [ ] HACK: at some point, the C++ code was producing collision object tsv 
	(line 198)
	
	<details>
	```{.python .numberLines startFrom="198"}
	# HACK: at some point, the C++ code was producing collision object tsv
	#     files with an extra tab put in, meaning there was an extra empty 
	#     string in every line when we split by tabs. hence, this line exists
	#     to deal with those files. it's safe to remove if you are only 
	#     dealing with freshly generated files
	```
	</details>

# **UGLY** -- 6 items
## [`../pyutil/eval_run.py`](../pyutil/eval_run.py) -- 1 item
 - [ ] UGLY: very fragile 
	(line 269)
	
	<details>
	```{.python .numberLines startFrom="269"}
	# UGLY: very fragile
	# get the angles and match with pairs
	
	```
	</details>

## [`../pyutil/genetic_utils.py`](../pyutil/genetic_utils.py) -- 2 items
 - [ ] UGLY: be able to modify the default fitness here 
	(line 636)
	
	<details>
	```{.python .numberLines startFrom="636"}
	# UGLY: be able to modify the default fitness here
	min_fitness : float = min([
	    fit
	    if not isnan(fit)
	    else 0.0
	```
	</details>

 - [ ] UGLY: this function should just call the continuation function after initialization 
	(line 792)
	
	<details>
	```{.python .numberLines startFrom="792"}
	# UGLY: this function should just call the continuation function after initialization
	def run_genetic_algorithm(
	        # for setup
	        rootdir : Path = "data/geno_sweep/",
	        dists : ModParamsDists = DEFAULT_DISTS,
	```
	</details>

## [`../pyutil/params.py`](../pyutil/params.py) -- 2 items
 - [ ] UGLY: this whole bit 
	(line 142)
	
	<details>
	```{.python .numberLines startFrom="142"}
	# UGLY: this whole bit
	if k.mod_type == ModTypes.conn.value:
	    _,str_from,str_to,_ = k.path.split(',')
	    k_write = f"{str_from}-{str_to}".replace('*','x')
	elif k.mod_type == ModTypes.params.value:
	```
	</details>

 - [ ] UGLY: clean this bit up 
	(line 201)
	
	<details>
	```{.python .numberLines startFrom="201"}
	# UGLY: clean this bit up
	if conn_key['to'].endswith('*'):
	    # if wildcard given, find every connection that matches
	    for nrn in params_data[conn_key['NS']]['neurons']:
	```
	</details>

## [`../pyutil/plot/gene.py`](../pyutil/plot/gene.py) -- 1 item
 - [ ] UGLY: this bit 
	(line 110)
	
	<details>
	```{.python .numberLines startFrom="110"}
	# UGLY: this bit
	cache_load : Callable[[],Dict[str,Any]] = lambda : raise_(NotImplementedError('check `scrape_extracted_cache()`'))
	cache_save : Callable[[Dict[str,float]],None] = lambda x : raise_(NotImplementedError('check `scrape_extracted_cache()`'))
	if format == 'msgpack':
	    import msgpack # type: ignore
	```
	</details>


