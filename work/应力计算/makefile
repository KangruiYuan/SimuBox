OBJ=ab2d_tensor_xx.o
ab2d_tensor_xx:$(OBJ)
	cc -O2 $(OBJ) -o ab2d_tensor_xx -lm -lfftw3 -lgsl -lgslcblas
ab2d_tensor_xx.o:ab2d_tensor_xx.c
	cc -O -c ab2d_tensor_xx.c