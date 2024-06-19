OBJ=babab.o
babab:$(OBJ)
	cc -O2 $(OBJ) -o babab -lm -lfftw3 -lgsl -lgslcblas
babab.o:babab.c
	cc -O -c babab.c	            
