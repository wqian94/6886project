OLEVEL ?= 3
CFLAGS += -std=c99 -g -fopenmp -O$(OLEVEL) -Wall
CCFLAGS += -std=c++17 -g -fopenmp -O$(OLEVEL) -Wall
CPP = g++

all: project

.PHONY: project
project: project.o
	$(CPP) $^ $(CFLAGS) -o $@
	chmod +x ./$@
	./$@


%.o: %.cc
	$(CPP) $(CCFLAGS) $*.cc -c

clean:
	rm -f project *.o
