# getting all sources and headers, for building docs
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

MODULES_HEADERS = $(wildcard **/*.h)
MODULES_SOURCES = $(wildcard **/*.cpp)

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
## GCCFLAGS = -pthread -c -O3 -flto
## GCCFLAGS = -std=c++11 -c -O3 -flto
# -flto is something to do with linking
GCCFLAGS = -std=c++17 -c -flto $(CFLAGS)

# building executables
singlerun: singlerun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o random.o Collide.o
	g++ $(CFLAGS) -o singlerun.exe singlerun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o random.o Collide.o

evolve: os evolve.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o
	@echo "this code is very possibly broken, and will probably be replaced by a python script"
	g++ $(CFLAGS) -o evolve.exe evolve.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o $(FLAG_PTHREAD)

demorun: demorun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o
	@echo "this is deprecated! dont use it!"
	g++ $(CFLAGS) -o demorun.exe demorun.o Worm.o WormBody.o NervousSystem.o StretchReceptor.o Muscles.o TSearch.o random.o Collide.o

# building modules
random.o: modules/random.cpp modules/random.h modules/VectorMatrix.h
	g++ $(GCCFLAGS) modules/random.cpp
TSearch.o: modules/TSearch.cpp modules/TSearch.h
	g++ $(GCCFLAGS) modules/TSearch.cpp
Worm.o: modules/Worm.cpp modules/Worm.h
	g++ $(GCCFLAGS) modules/Worm.cpp
Collide.o: modules/Collide.cpp modules/Collide.h
	g++ $(GCCFLAGS) modules/Collide.cpp
WormBody.o: modules/WormBody.cpp modules/WormBody.h modules/Collide.h
	g++ $(GCCFLAGS) modules/WormBody.cpp
NervousSystem.o: modules/NervousSystem.cpp modules/NervousSystem.h modules/VectorMatrix.h modules/random.h
	g++ $(GCCFLAGS) modules/NervousSystem.cpp
StretchReceptor.o: modules/StretchReceptor.cpp modules/StretchReceptor.h
	g++ $(GCCFLAGS) modules/StretchReceptor.cpp
Muscles.o: modules/Muscles.cpp modules/Muscles.h modules/VectorMatrix.h modules/random.h
	g++ $(GCCFLAGS) modules/Muscles.cpp
evolve.o: evolve.cpp modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	g++ $(GCCFLAGS) evolve.cpp
demorun.o: modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	g++ $(GCCFLAGS) demorun.cpp
singlerun.o: modules/Worm.h modules/WormBody.h modules/StretchReceptor.h modules/Muscles.h modules/TSearch.h modules/Collide.h
	g++ $(GCCFLAGS) singlerun.cpp

# misc
os:
	@echo "detected os: " $(detected_os)
	@echo "modified vars:" 
	@echo "    FLAG_PTHREAD: " $(FLAG_PTHREAD)

# cleaning up
clean:
	rm *.o *.exe

# building documentation
doc:
	@echo "Generating documentation..."; \
	cldoc generate $(GCCFLAGS) -- $(COMMON_DOC_FLAGS)

