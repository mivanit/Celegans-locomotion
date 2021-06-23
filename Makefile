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
    CFLAGS = -O3
endif

# compiler flags
CC = G++
## GCCFLAGS = -pthread -c -O3 -flto
## GCCFLAGS = -std=c++11 -c -O3 -flto
# -flto is something to do with linking
GCCFLAGS = -std=c++17 -c -flto $(CFLAGS)

# building executables
.PHONY: singlerun
singlerun: singlerun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o random.o Collide.o
	@echo "# [DEFAULT] Compiling executable for single worm sim"
	$(CC) $(CFLAGS) -o singlerun.exe singlerun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o random.o Collide.o

.PHONY: evolve
evolve: os evolve.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o
	@echo "# [DEPRECATED] Compiling genetic alg optimization"
	@echo "this code is very possibly broken, and will probably be replaced by a python script"
	$(CC) $(CFLAGS) -o evolve.exe evolve.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o $(FLAG_PTHREAD)

.PHONY: demorun
demorun: demorun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o
	@echo "# [DEPRECATED] Compiling demo run"
	@echo "this is deprecated! dont use it!"
	$(CC) $(CFLAGS) -o demorun.exe demorun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o

# building modules
random.o: modules/random.cpp modules/random.h modules/VectorMatrix.h
	$(CC) $(GCCFLAGS) modules/random.cpp
TSearch.o: modules/TSearch.cpp modules/TSearch.h
	$(CC) $(GCCFLAGS) modules/TSearch.cpp
Worm.o: modules/Worm.cpp modules/Worm.h
	$(CC) $(GCCFLAGS) modules/Worm.cpp
Collide.o: modules/Collide.cpp modules/Collide.h
	$(CC) $(GCCFLAGS) modules/Collide.cpp
WormBody.o: modules/WormBody.cpp modules/WormBody.h modules/Collide.h
	$(CC) $(GCCFLAGS) modules/WormBody.cpp
NervousSystem.o: modules/NervousSystem.cpp modules/NervousSystem.h modules/VectorMatrix.h modules/random.h
	$(CC) $(GCCFLAGS) modules/NervousSystem.cpp
StretchReceptor.o: modules/StretchReceptor.cpp modules/StretchReceptor.h
	$(CC) $(GCCFLAGS) modules/StretchReceptor.cpp
Muscles.o: modules/Muscles.cpp modules/Muscles.h modules/VectorMatrix.h modules/random.h
	$(CC) $(GCCFLAGS) modules/Muscles.cpp
evolve.o: evolve.cpp modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	$(CC) $(GCCFLAGS) evolve.cpp
demorun.o: modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	$(CC) $(GCCFLAGS) demorun.cpp
singlerun.o: modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	$(CC) $(GCCFLAGS) singlerun.cpp

# cleaning up
.PHONY: clean
clean:
	@echo "# cleaning up compiled files"
	-rm *.o *.exe $(PACKAGES_DIR)/*.gch

.PHONY: cleanob
cleanob:
	@echo "# cleaning up object files only"
	-rm *.o

.PHONY: clean_nogch
clean_nogch:
	@echo "# cleaning up compiled files, but leaving precompiled package headers"
	-rm *.o *.exe


# building documentation
.PHONY: doc
doc:
	@echo "# [WIP] Generating documentation"
	cldoc generate $(GCCFLAGS) -- $(COMMON_DOC_FLAGS)

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
	@echo "# NOTE: it is expected that 'singlerun.exe' has been compiled with 'PROFILE=1'"
	@echo "#       and run at least once, generating 'gmon.out'"
	@echo "# Writing analysis of 'gmon.out' file to 'prof.txt':"
	gprof singlerun.exe gmon.out > prof.txt

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
	$(foreach var,$(PACKAGES_HEADERS),$(CC) $(var);)



