# getting all sources and headers, for building docs
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

MODULES_HEADERS = $(wildcard **/*.h)
MODULES_SOURCES = $(wildcard **/*.cpp)

PACKAGES_DIR = modules/packages
PACKAGES_HEADERS = $(wildcard $(PACKAGES_DIR)/*.hpp)

COMMON_DOC_FLAGS = --report --output docs $(HEADERS) $(SOURCES) $(MODULES_HEADERS) $(MODULES_SOURCES)

# detecting os
ifeq ($(OS),Windows_NT)
	detected_os = Windows
	FLAG_PTHREAD = -lpthread
else
	detected_os = Linux
	FLAG_PTHREAD = -pthread
endif

# for building to use with GNUprof
PROFILE ?= 0
ifeq ($(PROFILE), 1)
    CFLAGS = -pg
else
    CFLAGS = -Ofast
endif

# compiler flags
CXX = g++
## GCCFLAGS = -pthread -c -O3 -flto
## GCCFLAGS = -std=c++11 -c -O3 -flto
## GCCFLAGS = -std=c++17 -c -flto $(CFLAGS)
# -flto is something to do with linking
GCCFLAGS = -std=c++17 $(CFLAGS)
MODULEFLAGS = -c -flto

# building executables
.PHONY: sim
sim: sim.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o random.o Collide.o
	@echo "# [DEFAULT] Compiling executable for worm sim"
	$(CXX) $(GCCFLAGS) -o sim.exe sim.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o random.o Collide.o

# TODO: read this https://stackoverflow.com/questions/1079832/how-can-i-configure-my-makefile-for-debug-and-release-builds
# TODO: implement output shortening flags

.PHONY: sim_Oshort
sim_Oshort: GCCFLAGS += -D_OUT_SHORT
sim_Oshort: sim
	@echo "# compiles sim with shortened output (head activations, head pos, no curve)"

.PHONY: sim_Omin
sim_Omin: GCCFLAGS += -D_OUT_MIN
sim_Omin: sim
	@echo "# compiles sim with minimum output (no activations, head pos, no curve)"

.PHONY: evolve
evolve: os evolve.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o
	@echo "# [DEPRECATED] Compiling genetic alg optimization"
	@echo "this code is very possibly broken, and will probably be replaced by a python script"
	$(CXX) $(GCCFLAGS) -o evolve.exe evolve.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o $(FLAG_PTHREAD)

.PHONY: demorun
demorun: demorun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o
	@echo "# [DEPRECATED] Compiling demo run"
	@echo "this is deprecated! dont use it!"
	$(CXX) $(GCCFLAGS) -o demorun.exe demorun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o

# building modules
random.o: modules/random.cpp modules/random.h modules/VectorMatrix.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/random.cpp
TSearch.o: modules/TSearch.cpp modules/TSearch.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/TSearch.cpp
Worm.o: modules/Worm.cpp modules/Worm.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/Worm.cpp
Collide.o: modules/Collide.cpp modules/Collide.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/Collide.cpp
WormBody.o: modules/WormBody.cpp modules/WormBody.h modules/Collide.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/WormBody.cpp
NervousSystem.o: modules/NervousSystem.cpp modules/NervousSystem.h modules/VectorMatrix.h modules/random.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/NervousSystem.cpp
StretchReceptor.o: modules/StretchReceptor.cpp modules/StretchReceptor.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/StretchReceptor.cpp
Muscles.o: modules/Muscles.cpp modules/Muscles.h modules/VectorMatrix.h modules/random.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) modules/Muscles.cpp
evolve.o: evolve.cpp modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) evolve.cpp
demorun.o: modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) demorun.cpp
sim.o: modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	$(CXX) $(GCCFLAGS) $(MODULEFLAGS) singlerun.cpp -o sim.o

# cleaning up
.PHONY: clean
clean:
	@echo "# cleaning up compiled files"
	-rm *.o *.exe $(PACKAGES_DIR)/*.gch

.PHONY: cleanob
cleanob:
	@echo "# cleaning up object files only"
	-rm *.o

.PHONY: cleangh
cleangh:
	@echo "# cleaning up compiled files, but leaving .gch precompiled package headers"
	-rm *.o *.exe


# building documentation
.PHONY: docs
docs:
	@echo "# [WIP] Generating documentation"
	pdoc pyutil/ -o ../docs/ --html --force
# cldoc generate $(GCCFLAGS) -- $(COMMON_DOC_FLAGS)

# python stuff
.PHONY: mypy
mypy:
	@echo "# run python static type checker"
	
	-mypy *.py pyutil/*.py


# misc
.PHONY: os
os:
	@echo "# showing detected operating system and resulting modified flags"
	@echo "# detected os: " $(detected_os)
	@echo "# modified vars:" 
	@echo "    FLAG_PTHREAD: " $(FLAG_PTHREAD)

.PHONY: prof
prof:
	@echo "# running profiling"
	@echo "# NOTE: it is expected that 'sim.exe' has been compiled with 'PROFILE=1'"
	@echo "#       and run at least once, generating 'gmon.out'"
	@echo "# Writing analysis of 'gmon.out' file to 'prof.txt':"
	gprof sim.exe gmon.out > prof.txt

# listing targets, from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
.PHONY: help
help:
	@echo -n "# Common make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 20

.PHONY: precomp
precomp:
	@echo "# Precompiling .gch files"
	$(foreach var,$(PACKAGES_HEADERS),$(CXX) $(var);)



