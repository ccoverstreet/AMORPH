# Put the directory above 'DNest4' into an environment variable
# called DNEST4_PATH for this to work.

CC = g++
CXXFLAGS = -std=c++14 -O3 -march=native -Wall -Wextra -pedantic -DNDEBUG
LIBS = -ldnest4 -lpthread -l:libyaml-cpp.a
#LIBS = -ldnest4 -lpthread -lyaml-cpp
#DNEST4_PATH=/home/brewer/Projects
DNEST4_PATH="../vendor"
LIB_YAMLCPP_INCLUDE="../vendor/yaml-cpp/include"
LIB_YAMLCPP_LIBPATH="../vendor/yaml-cpp/build"


default:
	make -C	../vendor/DNest4/code/
	mkdir -p ../vendor/yaml-cpp/build
	
	cd ../vendor/yaml-cpp/build && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
	make -C ../vendor/yaml-cpp/build

	$(CC) -I$(DNEST4_PATH) -I$(LIB_YAMLCPP_INCLUDE) -L$(DNEST4_PATH)/DNest4/code -L$(LIB_YAMLCPP_LIBPATH) $(CXXFLAGS)  -o AMORPH *.cpp $(LIBS)

windows:
	x86_64-w64-mingw32-g++ -I /usr/x86_64-w64-mingw32/include -I $(DNEST4_PATH) -std=c++11 -O3 -Wall -Wextra -pedantic -DNDEBUG -c *.cpp
	x86_64-w64-mingw32-g++ -static -pthread -o AMORPH.exe *.o $(DNEST4_PATH)/DNest4/code/dnest4.lib -lpthread -lyaml-cpp
	rm -f *.o
