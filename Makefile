#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES)
# $(INI_PARSER)
LIBRARIES = $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES)
CFLAGS = -Wall -fopenmp -march=native -O4 -Wshadow -fPIC

OBJECTS = lib/*.o

all:
	# make minIni
	mkdir -p lib
	$(GCC) src/random.c -c -o lib/random.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/grf.c -c -o lib/grf.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/fft.c -c -o lib/fft.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/peak.c -o peak $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)

	$(GCC) src/peak.c -o peak.so -shared $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

format:
	clang-format-10 -style="{BasedOnStyle: LLVM, IndentWidth: 4, AlignConsecutiveMacros: true, IndentPPDirectives: AfterHash}" -i src/*.c include/*.h
	astyle -i src/*.c include/*.h
