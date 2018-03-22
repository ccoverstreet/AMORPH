# Put the directory above 'DNest4' into an environment variable
# called DNEST4_PATH for this to work.

CC = g++
CXXFLAGS = -std=c++11 -Ofast -march=native -Wall -Wextra -pedantic -DNDEBUG
LIBS = -ldnest4 -lpthread

default:
	$(CC) -I$(DNEST4_PATH) $(CXXFLAGS) -c *.cpp
	$(CC) -pthread -L$(DNEST4_PATH)/DNest4/code -o main *.o $(LIBS)
	rm *.o

windows:
	x86_64-w64-mingw32-g++-posix -I$(DNEST4_PATH) $(CXXFLAGS) -c *.cpp
	x86_64-w64-mingw32-g++-posix -static  -pthread -L$(DNEST4_PATH)/DNest4/code -o AMORPH.exe *.o dnest4.lib
	rm -f *.o
