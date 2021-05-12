# Put the directory above 'DNest4' into an environment variable
# called DNEST4_PATH for this to work.

CC = g++
CXXFLAGS = -std=c++14 -O3 -march=native -Wall -Wextra -pedantic -DNDEBUG
LIBS = -ldnest4 -lpthread -lyaml-cpp
#DNEST4_PATH=/home/brewer/Projects
DNEST4_PATH="../vendor"

default:
	make -C	../vendor/DNest4/code/
	$(CC) -I$(DNEST4_PATH) $(CXXFLAGS) -c *.cpp
	$(CC) -pthread -L$(DNEST4_PATH)/DNest4/code -o AMORPH *.o $(LIBS)
	rm *.o

windows:
	x86_64-w64-mingw32-g++ -I /usr/x86_64-w64-mingw32/include -I $(DNEST4_PATH) -std=c++11 -O3 -Wall -Wextra -pedantic -DNDEBUG -c *.cpp
	x86_64-w64-mingw32-g++ -static -pthread -o AMORPH.exe *.o $(DNEST4_PATH)/DNest4/code/dnest4.lib -lpthread -lyaml-cpp
	rm -f *.o
