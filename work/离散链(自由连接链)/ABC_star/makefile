OBJ=abc_star_short.o
#Path=/usr/local
Libs=-lm -lfftw3_mpi -lfftw3 -L$(PATH)/gsl/lib -lgsl -lgslcblas -L${MKLROOT}/lib/intel64 -L${FFTWDIR}/lib
Includes=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread  -I$(PATH)/include -I${FFTWDIR}/include

abc_star_short:$(OBJ)
	mpicc -O3 -Wall -fopenmp $(OBJ) -o abc_star_short $(Libs)
	rm -rf *.o
  
abc_star_short.o:abc_star_short.c 
	mpicc -O3 -c -Wall -fopenmp abc_star_short.c $(Includes)
 

run16:
	rm -rf *.o  
	sqsub -f mpi -r 4h -N 1 -n 16 --mpp=30g -o output.out ./abc_star_short


1d:
	sqsub -f mpi -r 4h -N 1 -n 16 --mpp=30g -o output.out /home/xiej/surfactant/Short/CA=1.25/1d/abc_star_short

2d:
	sqsub -f mpi -r 4h -N 1 -n 16 --mpp=30g -o output.out /home/xiej/surfactant/Short/CA=1.25/2d/abc_star_short



test:
	cc -Wall -fopenmp -O2 -g $(Includes) ./abc_star_short.c $(Libs) -o test_prog
	valgrind --track-origins=yes --leak-check=full ./abc_star_short > errlog.txt 2>&1




clean:
	rm -rf output.out freeABW
	rm -rf *.dat *.txt *.log

