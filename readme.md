# Overview
A tool for simulating both the electrophysiology and mechanical body of a _C. Elegans_ nematode. Original code from:

> Izquierdo EJ, Beer RD. 2018 From head to tail: a neuromechanical model of forward locomotion in Caenorhabditis elegans. Phil. Trans. R. Soc. B 20170374. http://dx.doi.org/10.1098/rstb.2017.0374

Code heavily modified by Michael Ivanitskiy [@mivanit](https://github.com/mivanit)

# Building
Build with `make`. You can also first to `make precomp` to precompile headers. `make help` will tell some useful things.

`make clean` will remove out files, executables, and precompiled headers, whilst `make clean_nogch` will leave the precompiled headers.


# Running
`./singlerun.exe --help` will provide more info on running the program directly. The important parameters are:
 - `--params` needs a json file that provides network topology and other parameters
 - `--coll` needs a `.tsv`-ish file with collision objects. an empty file also works
examples of these can be found in the `input/` directory. generating collision object files is pretty painful, look in `pyutil/collision_object.py` for more on this.

`python multi_run.py --help` will provide info on launchers for running parameter sweeps or experiments with different food placement. This is probably a better way of running it. For help on a specific launcher, do `python multi_run.py LAUNCHER_NAME -- --help`


# Plotting/Analysis
Scripts of the form `pyutil/plot_*.py` will let you plot the position of the worm, make activation traces, or animations. do `--help` to see a list of functions in a script, or `FUNC_NAME -- --help` for help with a specific function.

# Profiling

build with `PROFILE=1`, run normally, and run analysis with `make prof` (simply calls gprof)
```bash
make singlerun PROFILE=1
./singlerun.exe [ARGS]
make prof
```


# Documentation
[ THIS DOESN'T WORK :( ]
to build, do
```bash
make doc
```
requirements:
 - make
 - [cldoc](https://jessevdk.github.io/cldoc)
 - [clang](https://clang.llvm.org), version 3.8+