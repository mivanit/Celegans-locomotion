
# profiling

build with `PROFILE=1`, run normally, and run analysis with `make prof` (simply calls gprof)
```bash
make singlerun PROFILE=1
./singlerun.exe [ARGS]
make prof
```


# Documentation

to build, do

```bash
make doc
```

requirements:
 - make
 - [cldoc](https://jessevdk.github.io/cldoc)
 - [clang](https://clang.llvm.org), version 3.8+