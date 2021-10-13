# master TODO list
## critical
- [x] fix bug with data not being saved
- [x] double check implementation of output level flags
- [x] fix `pyutil/plot/pos.py pos_multi`
- [ ] write & test angle fitness function
- [ ] fix code for running on cluster
- [ ] get diffusion stuff working
## org
- [x] fix and incorporate TODO scraper
- [ ] get docs working
- [ ] reorg source files into `src/` dir?
## params writing
 - [ ] save genetic run info to `params.json` when generating:
   - [ ] ModParams
   - [ ] genetic run id
   - [ ] generation id
   - [ ] individual id
   - [ ] eval runs params
## reading existing runs
 - [ ] error checking in validating params





# OLD

# 2021-07-24 17:04
## params writing
 - [ ] save genetic run info to `params.json` when generating:
   - [ ] ModParams
   - [ ] genetic run id
   - [ ] generation id
   - [ ] individual id
   - [ ] eval runs params
## reading existing runs
 - [x] move stuff from `util` and `plot_act` into `read_runs`
 - [x] name `extract_run_data` something more meaningful
 - [x] save into json/msgpack
 - [x] figure out how to deal nicely with mutliple runs per eval?
 - [ ] error checking in validating params





# 2021-06-28 22:25
look into this for writing data
https://forum.hdfgroup.org/t/write-simulation-data-line-by-line/4657/2

# 2021-06-24 02:00
https://www.desmos.com/calculator/3d3qkqzdmo

try $$e^{-x}$$ as a loss func

 
# 2021-05-27 13:38
connection keys can't be seamlessly turned into NamedTuples because `from` is a python keyword. So, at some point I should go in and change the 'from'/'to' to something else, and then the python code can be cleaned up a bit and made type checkable