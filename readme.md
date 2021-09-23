# Overview
A tool for simulating both the electrophysiology and mechanical body of a _C. Elegans_ nematode. Original code from:

> Izquierdo EJ, Beer RD. 2018 From head to tail: a neuromechanical model of forward locomotion in Caenorhabditis elegans. Phil. Trans. R. Soc. B 20170374. http://dx.doi.org/10.1098/rstb.2017.0374

Code heavily modified by Michael Ivanitskiy [@mivanit](https://github.com/mivanit)

Code about to be modified by Yash Mehta. First commit. Second commit.

# Building
Build with `make` (calls `make sim`)

- `make help` lists targets with descriptions
- if you recompile the code a lot, run `make precomp` to precompile headers
- `make clean` will remove out files, executables, and precompiled headers
  - `make clean_nogch` will leave the precompiled headers


# Running
## Running executable directly
`./sim.exe --help` will provide more info on running the program directly. The important parameters are:
 - `--params` needs a json file that provides network topology and other parameters
 - `--coll` needs a `.tsv`-ish file with collision objects. an empty file also works

examples of these can be found in the `input/` directory. generating collision object files is pretty painful, look in `pyutil/collision_object.py` for more on this.

## Using python launchers
`python multi_run.py --help` will provide a list of launchers for running parameter sweeps or experiments with different food placement. This is generally a more useful way of running the code. 

For help on a specific launcher, do `python multi_run.py LAUNCHER_NAME -- --help`

## parameter optimization
Work in progress -- check `optimize_params.py`


# Plotting/Analysis
Scripts of the form `pyutil/plot_*.py` will let you plot the position of the worm, make activation traces, or animations. do `python pyutil/plot_something.py --help` to see a list of functions in a script, or `python pyutil/plot_something.py FUNC_NAME -- --help` for help with a specific function.

**NOTE:** these must be run from the root directory, not from inside `pyutil/`. This makes referencing paths to `data/run/` slightly easier, but also makes the python imports work

# Profiling

build with `PROFILE=1`, run normally (a `gmon.out` file should be generated), and run analysis with `make prof` (simply calls gprof):

```bash
make sim PROFILE=1
./sim.exe [ARGS]
make prof
```

A file called `prof.txt` with useful info should be generated. You can also look in [`data/prof/prof.txt`](data/prof/prof.txt) for an already generated file. If you optimize the code, label the new `prof.txt` with a commit hash or date.


# Documentation
documentation for the python code can be build automatically using [`pdoc`](https://pdoc3.github.io/pdoc/)
to build, do

```bash
make docs
```

Whenever I can get automatic generation of documentation for the C++ code to work, it will also be launched using `make docs`


# Randomization analysis instructions

- create a directory `data/randomruns` (call it whatever you like)
- run the command `./sim.exe --params input/params.json --output data/randomruns/run_n --rand`, replacing `run_n` with whichever run you are doing
  - you can also do `./sim.exe --params input/params.json --output data/randomruns/run_n --seed <N>` if you want to specify a seed by changing `<N>` to a number
- to plot a single run, do `python pyutil/plot/pos.py pos data/randomruns/run_n`
- to plot multiple runs, do `python pyutil/plot/pos.py pos_multi --rootdir=data/randomruns/run_n`
  - this is currently broken -- look for the "TODO" in the code and see if you can fix it

<!-- requirements:

 - make
 - [cldoc](https://jessevdk.github.io/cldoc)
 - [clang](https://clang.llvm.org), version 3.8+ -->