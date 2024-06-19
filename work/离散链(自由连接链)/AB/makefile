OBJ=ab_short_removeMPI.o
ab_short_removeMPI:$(OBJ)
	cc -O2 $(OBJ) -o ab_short_removeMPI -lm -lfftw3 -lgsl -lgslcblas
ab_short_removeMPI.o:ab_short_removeMPI.c
	cc -O -c ab_short_removeMPI.c            
