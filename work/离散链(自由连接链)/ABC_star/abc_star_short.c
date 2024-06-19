/************************************************************************/
/* 
	ABC star copolymer system modeled by freely-jointed chain with finite
	range interaction

	Variable cell shape optimization (nd = 6 or 3)

	Accelerated by MPI
*/
/************************************************************************/



#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<mpi.h>
#include<fftw3-mpi.h>
#include<fftw3.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_linalg.h>


#define MaxIT 2000        //Maximum iteration steps

#define sigma blA
#define blA 1.0
#define blB (1.0/2.0)
#define blC (1.0/2.0)

#define nd 6
#define nb_max 3

#define Pi 3.141592653589

// Parameters used in Anderson convergence //
#define N_hist 30
//

void init_Reading(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC);
void init_Random_primitive(double *phi_min, double *phi_maj);
void init_Disordered_primitive(double *phi_min, double *phi_maj);
void init_Lamella_primitive(double *phi_min, double *phi_maj);
void init_HEX_primitive(double *phi_min, double *phi_maj);
void init_BCC_primitive(double *phi_min, double *phi_maj);
void init_FCC_primitive(double *phi_min, double *phi_maj);
void init_HCP_primitive(double *phi_min, double *phi_maj);
void init_Gyroid_primitive(double *phi_min, double *phi_maj);
void init_DoubleGyroid_primitive(double *phi_min, double *phi_maj);
void init_Diamond_primitive(double *phi_min, double *phi_maj);
void init_DoubleDiamond_primitive(double *phi_min, double *phi_maj);
void init_C14_primitive(double *phi_min, double *phi_maj);
void init_C15_primitive(double *phi_min, double *phi_maj);
void init_A15_primitive(double *phi_min, double *phi_maj);
void init_Sigma_primitive(double *phi_min, double *phi_maj);
void init_Z_primitive(double *phi_min, double *phi_maj);
void init_O70_primitive(double *phi_min, double *phi_maj);
void init_DoublePlumberNightmare_primitive(double *phi_min, double *phi_maj);

void finalize_initialization(double *phA,double *phB,double *phC,int inv);
void init_w_from_phi(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC);
void init_lattice_parameters();
void discretize_space();

struct result_struct freeE(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC);
double getConc(double *phA, double *phB, double *phC, double *wA, double *wB, double *wC);
void get_q(double *prop, fftw_complex *props, double *exp_w, double *g_k, double *qInt, int ns, int sign);
void conjunction_propagation(double *qInt, fftw_complex *qInts, double *g_k);
void branching_propagation(int num_branches, double *qInt, fftw_complex *qInts_n[], double *g_k_n[], double *g_k, double *exp_w, double *qJoint, fftw_complex *qJoints, int idxJoint, int IF_STORE, int idxStore, double *qBranch);
void cal_B_matrix();
void cal_k_vectors();
void cal_der_k2();
void forward_fft_mpi(double *f_r, fftw_complex *f_k);
void backward_fft_mpi(fftw_complex *f_k, double *f_r);

void write_ph(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC);
void write_phA(double *phA);
void write_phB(double *phB);
void write_phC(double *phC);
void write_ph_init_point(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC);
void reading_init_point(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC);
int output(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC,double scan_para,struct result_struct results,char filename[]);

double error_cal(double *waDiffs, double *wbDiffs, double *wcDiffs, double *wAs, double *wBs, double *wCs);
void update_flds_hist(double *waDiff, double *wbDiff, double *wcDiff, double *wAnew, double *wBnew, double *wCnew, double *del, double *outs);
void Anderson_mixing(double *del, double *outs, int N_rec, double *wA, double *wB, double *wC);

void Distribute(double *wA,double *wB,double *wC);
void Gather(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC);


struct result_struct
{
	double freeEnergy, inCompMax, freeDiff, err, stress_max;
};


//*****Global variables*****//
// Parameters of the polymer chain:
int NsA, NsB, NsC, N; 
double chiAB, chiAC, chiBC;   
// Target Phase:
int in_Phase;
double fix_para[3];
// Period and box size:
double *A_matrix, *k_square, *k_norm, *u_k, *sum_factor;
double *h_idx, *k_idx, *l_idx;
double *dk2_da, *dk2_db, *dk2_dc, *dk2_dd, *dk2_de, *dk2_df;
int Na, Nb, Nc;
int local_Na_2_NbNc, local_NaNbNc1, Na_2_NbNc;
int NaNbNc, Nah1, NaNbNc1;
int hist_Na;
double lattice_para[nd],lattice_para_new[nd];
// Initialization parameters:
double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
// FFTW:
double *in;
fftw_complex *out, *deltaf;
fftw_plan p_forward, p_backward;
// Stress:
double *stress,*stress_bond,*stress_FR;
// Time test:
double start, end;
// MPI:
int world_rank, world_size;
ptrdiff_t alloc_local, local_Nc, local_Nc_start;

int main(int argc, char **argv)
{
	double *wA,*wB,*wC,*phA,*phB,*phC;
	struct result_struct e1;
	int i,j,k,iseed=-3; 
	int in_method;
	long ijk;
	//Inverse phase?
	int inverse;
	// Scan parameters:
	double change1, change2;
	int times, double_direction;
	// Output file naming:
	char filename[100], phasename[100], category[100];

	FILE *fp;
	time_t ts;
	iseed=time(&ts);
	srand48(iseed);

	// Initialize the MPI environment
  	MPI_Init(&argc, &argv);
	// Find out rank, size
 	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  	A_matrix=(double *)calloc(9,sizeof(double));

	//*****Read in parameters from a file named para*****//
	fp=fopen("./../para_common","r");
	fscanf(fp,"%d", &inverse);		
	fscanf(fp,"%d,%d,%d",&NsA, &NsB, &NsC);
	fclose(fp);

	fp=fopen("para","r");
	fscanf(fp,"%d,%d",&in_method, &in_Phase);
	fscanf(fp,"%lf,%lf,%lf", &chiAB, &chiAC, &chiBC);	
	fscanf(fp,"%lf,%d",&change1,&times);
	fscanf(fp,"%d,%lf",&double_direction,&change2);
  	fscanf(fp,"%lf",&A_matrix[0]);			
  	fclose(fp);

  	//*****Set the name of the output file*****//
  	//*****Determine the category of target phase*****//
  	if(inverse==0)
  	{
  		sprintf(category, "");
  	}
  	else
  	{
  		sprintf(category, "inv_");
  	}

  	//*****Determine the name of the target phase*****//
  	if(in_Phase==0)
	{
  		sprintf(phasename, "Dis");
  		sprintf(category, "");
	}
	else if(in_Phase==1)
	{
  		sprintf(phasename, "Lam");
	}
	else if(in_Phase==2)
	{
  		sprintf(phasename, "HEX");
	}
	else if(in_Phase==3)
	{
  		sprintf(phasename, "BCC");
	}    
	else if(in_Phase==4)
	{
  		sprintf(phasename, "FCC");
	}  
	else if(in_Phase==5)
	{
  		sprintf(phasename, "HCP");
	} 
	else if(in_Phase==6)
	{
  		sprintf(phasename, "G");
	} 
	else if(in_Phase==7)
	{
  		sprintf(phasename, "DG");
	} 
	else if(in_Phase==8)
	{
		sprintf(phasename, "D");
	}
	else if(in_Phase==9)
	{
  		sprintf(phasename, "DD");
	}   
	else if(in_Phase==10)
	{
  		sprintf(phasename, "C14");
	} 
	else if(in_Phase==11)
	{
  		sprintf(phasename, "C15");
	} 
	else if(in_Phase==12)
	{
  		sprintf(phasename, "A15");
	} 
	else if(in_Phase==13)
	{
  		sprintf(phasename, "Sigma");
	}  
	else if(in_Phase==14)
	{
  		sprintf(phasename, "Z");
	}   
	else if(in_Phase==15)
	{
  		sprintf(phasename, "O70");
	}   
	else if(in_Phase==16)
	{
  		sprintf(phasename, "DPN");
	}  
	else
	{
		sprintf(phasename, "Random");
		sprintf(category, "");
	}

  	//*****Generate the name of the output file*****//
  	sprintf(filename, "%s%s.dat", category, phasename);

  	//*****Set the number of grid points according to the target phase*****//
	discretize_space();

	/* Assign values for looping constants */
	NaNbNc = Na*Nb*Nc;
	Nah1 = Na/2+1;
	NaNbNc1 = Nc*Nb*Nah1;

	/* Initialize MPI FFTW */
  	fftw_mpi_init();

  	/* get local data size and allocate */
  	alloc_local = fftw_mpi_local_size_3d(Nc, Nb, Nah1, MPI_COMM_WORLD, &local_Nc, &local_Nc_start);

  	/* Assign values for looping constants needed by MPI */
  	Na_2_NbNc = (Na+2)*Nb*Nc;   // Add padding space in real space used by MPI FFTW
	local_Na_2_NbNc = (Na+2)*Nb*local_Nc;
	local_NaNbNc1 = Nah1*Nb*local_Nc;
	hist_Na = local_Na_2_NbNc*3+nd;

  	in = fftw_alloc_real(2 * alloc_local);
  	out = fftw_alloc_complex(alloc_local);

  	/* create plan for out-of-place r2c DFT */
  	p_forward = fftw_mpi_plan_dft_r2c_3d(Nc, Nb, Na, in, out, MPI_COMM_WORLD, FFTW_MEASURE);
  	p_backward = fftw_mpi_plan_dft_c2r_3d(Nc, Nb, Na, out, in, MPI_COMM_WORLD, FFTW_MEASURE);

  	h_idx=(double *)calloc(Na,sizeof(double));
	k_idx=(double *)calloc(Nb,sizeof(double));
	l_idx=(double *)calloc(Nc,sizeof(double));
  	stress=(double *)malloc(sizeof(double)*nd);
  	stress_bond=(double *)malloc(sizeof(double)*nd);
  	stress_FR=(double *)malloc(sizeof(double)*nd);
  	k_square=(double *)malloc(sizeof(double)*local_NaNbNc1);
  	k_norm=(double *)malloc(sizeof(double)*local_NaNbNc1);
  	dk2_da=(double *)malloc(sizeof(double)*local_NaNbNc1);
  	dk2_db=(double *)malloc(sizeof(double)*local_NaNbNc1);
  	dk2_dc=(double *)malloc(sizeof(double)*local_NaNbNc1);
  	dk2_dd=(double *)malloc(sizeof(double)*local_NaNbNc1);
  	dk2_de=(double *)malloc(sizeof(double)*local_NaNbNc1);
  	dk2_df=(double *)malloc(sizeof(double)*local_NaNbNc1);
	u_k=(double *)malloc(sizeof(double)*local_NaNbNc1);
	sum_factor=(double *)malloc(sizeof(double)*local_NaNbNc1);
	deltaf=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);
  	if(world_rank == 0) 
  	{
		wA=(double *)malloc(sizeof(double)*Na_2_NbNc);
		wB=(double *)malloc(sizeof(double)*Na_2_NbNc);
		wC=(double *)malloc(sizeof(double)*Na_2_NbNc);
		phA=(double *)malloc(sizeof(double)*Na_2_NbNc);
		phB=(double *)malloc(sizeof(double)*Na_2_NbNc);
		phC=(double *)malloc(sizeof(double)*Na_2_NbNc);
	}
	else
	{
		wA=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
		wB=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
		wC=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
		phA=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
		phB=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
		phC=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	}

	//Indexes used to define the k-vectors//
  	for(i=0;i<Na/2;i++)h_idx[i]=i*1.0;  
	for(i=Na/2;i<Na;i++)h_idx[i]=(i-Na)*1.0;
  
	for(i=0;i<Nb/2;i++)k_idx[i]=i*1.0;  
	for(i=Nb/2;i<Nb;i++)k_idx[i]=(i-Nb)*1.0;
  
 	for(i=0;i<Nc/2;i++)l_idx[i]=i*1.0;  
  	for(i=Nc/2;i<Nc;i++)l_idx[i]=(i-Nc)*1.0;

  	// delta(k)
	for(ijk=0; ijk<local_NaNbNc1;ijk++)
	{
		deltaf[ijk][0] = 0.0;
		deltaf[ijk][1] = 0.0;
	}
	if(world_rank==0)
	{
		deltaf[0][0] = 1.0;
	}

	//Used to do the summation over all components in Fourier space//
	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(k*Nb+j)*Nah1+i;
		if(i==0||i==Nah1-1)
		{
			sum_factor[ijk]=1.0;
		}
		else
		{
			sum_factor[ijk]=2.0;
		}
	} 

  	//Initialize lattice parameters according to the target phase//
  	init_lattice_parameters();

  	N=NsA+NsB+NsC+1;

	if(world_rank == 0) 
  	{
		if(in_method==0)
	  	{    
	    	if(in_Phase==0)
	    	{
	      		init_Disordered_primitive(phA,phB);
	    	}
	    	else if(in_Phase==1)
	    	{
	      		init_Lamella_primitive(phA,phB);
	    	}
	    	else if(in_Phase==2)
	    	{
	      		init_HEX_primitive(phA,phB);
	    	}
	    	else if(in_Phase==3)
	    	{
	      		init_BCC_primitive(phA,phB);
	    	}    
	    	else if(in_Phase==4)
	    	{
	      		init_FCC_primitive(phA,phB);
	    	}  
	    	else if(in_Phase==5)
	    	{
	      		init_HCP_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==6)
	    	{
	      		init_Gyroid_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==7)
	    	{
	      		init_DoubleGyroid_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==8)
	    	{
	    		init_Diamond_primitive(phA,phB);
	    	}
	    	else if(in_Phase==9)
	    	{
	      		init_DoubleDiamond_primitive(phA,phB);
	    	}   
	    	else if(in_Phase==10)
	    	{
	      		init_C14_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==11)
	    	{
	      		init_C15_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==12)
	    	{
	      		init_A15_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==13)
	    	{
	      		init_Sigma_primitive(phA,phB);
	    	}
	    	else if(in_Phase==14)
	    	{
	      		init_Z_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==15)
	    	{
	      		init_O70_primitive(phA,phB);
	    	} 
	    	else if(in_Phase==16)
	    	{
	      		init_DoublePlumberNightmare_primitive(phA,phB);
	    	}  
	    	else
	    	{
	    		init_Random_primitive(phA,phB);
	    		printf("Random initialization is used !");
	    	} 

	    	finalize_initialization(phA,phB,phC,inverse);

	    	init_w_from_phi(wA,wB,wC,phA,phB,phC);
	  	}   
		else if(in_method==1)
		{ 
	    	init_Reading(wA,wB,wC,phA,phB,phC);   
	  	}
	}

	Distribute(wA,wB,wC);

	// // Check Distribute()
	// if(world_rank == 0)
	// {
	// 	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	// 	{
	// 		ijk=(k*Nb+j)*(Na+2)+i;
	// 		printf("%d : %lf\n", world_rank, wA[ijk]);
	// 	}
	// }
	// else
	// {
	// 	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	// 	{
	// 		ijk=(k*Nb+j)*(Na+2)+i;
	// 		printf("%d : %lf\n", world_rank, wA[ijk]);
	// 	}
	// } 

	int diagnosis, count=0;
	chiAC=chiAB;

	double init_point=chiAB, count_init_point, A_matrix_init_point[9];
	
	// Calculate the init point
	e1=freeE(wA,wB,wC,phA,phB,phC);

  	Gather(phA,phB,phC,wA,wB,wC);

  	diagnosis=output(phA,phB,phC,wA,wB,wC,chiAB,e1,filename);

  	chiAB+=change1;
	chiAC+=change1;

  	if(diagnosis==0)
  	{
  		count+=1;
  	}

  	// Store the result of the init point
  	count_init_point=count;

  	for(i=0;i<9;i++)
  	{
  		A_matrix_init_point[i]=A_matrix[i];
  	}

  	if(world_rank==0)
  	{
  		write_ph_init_point(phA,phB,phC,wA,wB,wC);
  	}
	
	// Forward scan
	for(i=0;i<times-1;i++)
  	{
  		if(count>=2)
	  	{
	  		break;
	  	}

	  	e1=freeE(wA,wB,wC,phA,phB,phC);

	  	Gather(phA,phB,phC,wA,wB,wC);

	  	diagnosis=output(phA,phB,phC,wA,wB,wC,chiAB,e1,filename);

	  	chiAB+=change1;
	  	chiAC+=change1;

	  	if(diagnosis==0)
	  	{
	  		count+=1;
	  	}
	}

	if(double_direction!=0)
	{
		// Restore the result of init point
		count=count_init_point;

		for(i=0;i<9;i++)
	  	{
	  		A_matrix[i]=A_matrix_init_point[i];
	  	}

	  	if(world_rank==0)
	  	{
	  		reading_init_point(wA,wB,wC,phA,phB,phC);
	  	}
	  	Distribute(wA,wB,wC);

		chiAB=init_point;
		chiAB+=change2;
		chiAC=chiAB;

		// Backward scan
		for(i=0;i<times-1;i++)
	  	{
	  		if(count>=2||chiAB<0.0)
		  	{
		  		break;
		  	}
		  	
		  	e1=freeE(wA,wB,wC,phA,phB,phC);

		  	Gather(phA,phB,phC,wA,wB,wC);

		  	diagnosis=output(phA,phB,phC,wA,wB,wC,chiAB,e1,filename);

		  	chiAB+=change2;
		  	chiAC+=change2;

		  	if(diagnosis==0)
		  	{
		  		count+=1;
		  	}
		}
	}

  	fftw_destroy_plan(p_forward);
  	fftw_destroy_plan(p_backward);

  	MPI_Finalize();
  	
  	fftw_free(in);
  	fftw_free(out);

  	free(A_matrix);
	free(wA);
	free(wB);
	free(wC);
	free(phA);
	free(phB);
	free(phC);
	free(h_idx);
	free(k_idx);
	free(l_idx);
  	free(k_square);	
  	free(k_norm);
  	free(dk2_da);
  	free(dk2_db);
  	free(dk2_dc);
  	free(dk2_dd);
  	free(dk2_de);
  	free(dk2_df);
	free(u_k);
	free(sum_factor);
	free(stress);
	free(stress_bond);
	free(stress_FR);
	fftw_free(deltaf);

	return 1;
}


//*************************************main loop****************************************
struct result_struct freeE(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC)
{
	int i,j,k,iter,maxIter;
	long ijk;
	double freeEnergy,freeOld,freeDiff,parfQ;
	double enthalpic, entropic, enthalpic_im, entropic_im, enthalpicAB, enthalpicAC, enthalpicBC, enthalpicAB_im, enthalpicAC_im, enthalpicBC_im;
	double Sm1,Sm2,Sm3,Sm4,wopt,psum,fpsum,lambda0,lambda;
	double *waDiff,*wbDiff,*wcDiff,inCompMax;
  	double *del, *outs, *wAnew, *wBnew, *wCnew, err, *dudk;
  	double *stress_FR_common, stress_max;
	int N_rec;
  	fftw_complex *phAs,*phBs,*phCs,*wAs,*wBs,*wCs,*wAnews,*wBnews,*wCnews,*etas;
  	FILE *fp;

  	waDiff=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	wbDiff=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	wcDiff=(double *)malloc(sizeof(double)*local_Na_2_NbNc);

 	phAs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);
	phBs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);
	phCs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);

	wAs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);
	wBs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);
	wCs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);
	wAnew=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	wBnew=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	wCnew=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	wAnews=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);
	wBnews=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);  
	wCnews=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1); 

	etas=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);

	del = (double *)malloc(sizeof(double)*N_hist*hist_Na);
	outs = (double *)malloc(sizeof(double)*N_hist*hist_Na);

	dudk=(double *)malloc(sizeof(double)*local_NaNbNc1);

	stress_FR_common=(double *)malloc(sizeof(double)*local_NaNbNc1);

  	Sm1=1e-7;
  	Sm2=1e-8;
  	Sm3=1e-7;
  	Sm4=1e-7;
  	maxIter=MaxIT;
  	wopt=0.05;
  	psum=0.0;
  	fpsum=0.0;
  	lambda0=1.0;

  	iter=0;	

	freeEnergy=0.0;

	lattice_para[0]=A_matrix[0];   //a
	lattice_para[1]=A_matrix[4];   //b
	lattice_para[2]=A_matrix[8];   //c
	lattice_para[3]=A_matrix[3];   //d
	lattice_para[4]=A_matrix[7];   //e
	lattice_para[5]=A_matrix[6];   //f

	if(chiAB!=chiAC)
	{
		fp=fopen("para_error","w");
    	fprintf(fp,"xAB and xAC are not equal in core %d!\n",world_rank);
		exit(0);
	}

	do
	{
		if(iter>=1200&&inCompMax>=1e-2)
		{
			break;
		}

		if(world_rank == 0) 
	  	{
	  		start=MPI_Wtime();
	  	}
		iter=iter+1;

		lambda=10.0*(1.0-lambda0);
    	lambda0*=0.999;

		A_matrix[0]=lattice_para[0];
		A_matrix[4]=lattice_para[1];
		A_matrix[8]=lattice_para[2];
		A_matrix[3]=fix_para[0]*A_matrix[4];
		A_matrix[7]=fix_para[1]*A_matrix[8];
		A_matrix[6]=fix_para[2]*A_matrix[8];

		//Define Fourier components//
		cal_k_vectors();
		cal_der_k2();

		for(ijk=0;ijk<local_NaNbNc1;ijk++)
		{
			u_k[ijk] = exp(-k_square[ijk]*sigma*sigma/6.0);
		}

	//1.Calculate propagators and then the segment concentrations//
    	parfQ=getConc(phA,phB,phC,wA,wB,wC);

    //2.Calculate incompresibility and inCompMax//
    	inCompMax=0.0;
		for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);
			psum=1.0-phA[ijk]-phB[ijk]-phC[ijk];			
			fpsum=fabs(psum);
			if(fpsum>inCompMax)inCompMax=fpsum;
		}

		MPI_Allreduce(MPI_IN_PLACE, &inCompMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	//3.Fourier transferm phi and w to k-space//
		forward_fft_mpi(phA, phAs);
		forward_fft_mpi(phB, phBs);
		forward_fft_mpi(phC, phCs);
		forward_fft_mpi(wA, wAs);
		forward_fft_mpi(wB, wBs);
		forward_fft_mpi(wC, wCs);

	//4.Update the presure field in k-space//		
		for(ijk=0; ijk<local_NaNbNc1;ijk++)
		{
			etas[ijk][0]=(wAs[ijk][0]+wBs[ijk][0]+wCs[ijk][0]-N*u_k[ijk]*(chiAB*(deltaf[ijk][0]-phCs[ijk][0])+chiBC*(deltaf[ijk][0]-phAs[ijk][0])+chiAC*(deltaf[ijk][0]-phBs[ijk][0])))/3.0-lambda*(deltaf[ijk][0]-phAs[ijk][0]-phBs[ijk][0]-phCs[ijk][0]);
			etas[ijk][1]=(wAs[ijk][1]+wBs[ijk][1]+wCs[ijk][1]-N*u_k[ijk]*(chiAB*(deltaf[ijk][1]-phCs[ijk][1])+chiBC*(deltaf[ijk][1]-phAs[ijk][1])+chiAC*(deltaf[ijk][1]-phBs[ijk][1])))/3.0-lambda*(deltaf[ijk][1]-phAs[ijk][1]-phBs[ijk][1]-phCs[ijk][1]);
		}  	

	//5.Obtain the output conjugate fields in k-space//
    	for(ijk=0;ijk<local_NaNbNc1;ijk++)
		{
      		wAnews[ijk][0]=N*u_k[ijk]*(chiAB*phBs[ijk][0]+chiAC*phCs[ijk][0])+etas[ijk][0];		
      		wAnews[ijk][1]=N*u_k[ijk]*(chiAB*phBs[ijk][1]+chiAC*phCs[ijk][1])+etas[ijk][1];
      		wBnews[ijk][0]=N*u_k[ijk]*(chiAB*phAs[ijk][0]+chiBC*phCs[ijk][0])+etas[ijk][0];		
      		wBnews[ijk][1]=N*u_k[ijk]*(chiAB*phAs[ijk][1]+chiBC*phCs[ijk][1])+etas[ijk][1];
      		wCnews[ijk][0]=N*u_k[ijk]*(chiAC*phAs[ijk][0]+chiBC*phBs[ijk][0])+etas[ijk][0];		
      		wCnews[ijk][1]=N*u_k[ijk]*(chiAC*phAs[ijk][1]+chiBC*phBs[ijk][1])+etas[ijk][1];
    	} 	

    //6.Backward Fourier transferm to obtain the output conjugate field in real space//
		backward_fft_mpi(wAnews, wAnew);
		backward_fft_mpi(wBnews, wBnew);
		backward_fft_mpi(wCnews, wCnew);
		
	//7.Compute the deviation functions or residuals//
		for(ijk=0; ijk<local_Na_2_NbNc;ijk++)
		{
			waDiff[ijk]=wAnew[ijk]-wA[ijk];
			wbDiff[ijk]=wBnew[ijk]-wB[ijk];
			wcDiff[ijk]=wCnew[ijk]-wC[ijk];
		}		

	//Stress//
		for(ijk=0; ijk<local_NaNbNc1;ijk++)
		{
			dudk[ijk]=-(sigma*sigma)*u_k[ijk]/6.0;
		}			

		for(ijk=0; ijk<local_NaNbNc1; ijk++)
		{
			stress_FR_common[ijk]=sum_factor[ijk]*dudk[ijk]*N*(chiAB*(phAs[ijk][0]*phBs[ijk][0]+phAs[ijk][1]*phBs[ijk][1])+
															   chiAC*(phAs[ijk][0]*phCs[ijk][0]+phAs[ijk][1]*phCs[ijk][1])+
															   chiBC*(phBs[ijk][0]*phCs[ijk][0]+phBs[ijk][1]*phCs[ijk][1]));
		} 

		for(i=0;i<nd;i++)
		{
			stress_FR[i]=0.0;
		}
		for(ijk=0; ijk<local_NaNbNc1; ijk++)
		{
			stress_FR[0]+=dk2_da[ijk]*stress_FR_common[ijk];
			stress_FR[1]+=dk2_db[ijk]*stress_FR_common[ijk];
			stress_FR[2]+=dk2_dc[ijk]*stress_FR_common[ijk];
			stress_FR[3]+=dk2_dd[ijk]*stress_FR_common[ijk];
			stress_FR[4]+=dk2_de[ijk]*stress_FR_common[ijk];
			stress_FR[5]+=dk2_df[ijk]*stress_FR_common[ijk];
		} 		    

		for(i=0;i<nd;i++)
		{
			stress[i]=-(-stress_bond[i]+stress_FR[i]);
		}
    	MPI_Allreduce(MPI_IN_PLACE, stress, nd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for(i=0;i<nd;i++)
		{
			lattice_para_new[i]=lattice_para[i]+stress[i]*sqrt(A_matrix[0]*A_matrix[4]*A_matrix[8]);
		}

	//8.Apply either simple mixing or Anderson mixing//
    	//judge the error
		err = error_cal(waDiff, wbDiff, wcDiff, wA, wB, wC);
    	//update the history fields, and zero is new fields
    	update_flds_hist(waDiff, wbDiff, wcDiff, wAnew, wBnew, wCnew, del, outs);

	  	if(iter<150)
	  	{	  
			for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
			{
				wA[ijk]+=wopt*waDiff[ijk];
		 		wB[ijk]+=wopt*wbDiff[ijk];
		 		wC[ijk]+=wopt*wcDiff[ijk];
			}

			for(i=0;i<nd;i++)
			{
				lattice_para[i]+=wopt*stress[i]*sqrt(A_matrix[0]*A_matrix[4]*A_matrix[8]);
			}
	  	}
	  	else
	  	{
	  		if(world_rank == 0)
	  		{
		    	if(iter==1||iter%10==0||iter>=maxIter)
		    	{
		    		FILE *fp=fopen("ing.dat","a");
		    		fprintf(fp, "/***** enter Anderson mixing *****/\n");
		    		fclose(fp);
		    	}
		    }
	  		N_rec = (iter-1)<N_hist?(iter-1):N_hist;
	  		Anderson_mixing(del, outs, N_rec, wA, wB, wC);
	  	}

	//9.Calculate the free energy density in k space//	

		enthalpicAB=0.0;
		enthalpicAC=0.0;
		enthalpicBC=0.0;
		entropic=0.0;
		enthalpicAB_im=0.0;
		enthalpicAC_im=0.0;
		enthalpicBC_im=0.0;
		entropic_im=0.0;

		for(ijk=0;ijk<local_NaNbNc1;ijk++)
    	{
			enthalpicAB=enthalpicAB+sum_factor[ijk]*N*u_k[ijk]*(chiAB*(phAs[ijk][0]*phBs[ijk][0]+phAs[ijk][1]*phBs[ijk][1]));

			enthalpicAC=enthalpicAC+sum_factor[ijk]*N*u_k[ijk]*(chiAC*(phAs[ijk][0]*phCs[ijk][0]+phAs[ijk][1]*phCs[ijk][1]));

			enthalpicBC=enthalpicBC+sum_factor[ijk]*N*u_k[ijk]*(chiBC*(phBs[ijk][0]*phCs[ijk][0]+phBs[ijk][1]*phCs[ijk][1]));

      		enthalpicAB_im=enthalpicAB_im+sum_factor[ijk]*N*u_k[ijk]*(chiAB*(phAs[ijk][1]*phBs[ijk][0]-phAs[ijk][0]*phBs[ijk][1]));

      		enthalpicAC_im=enthalpicAC_im+sum_factor[ijk]*N*u_k[ijk]*(chiAC*(phAs[ijk][1]*phCs[ijk][0]-phAs[ijk][0]*phCs[ijk][1]));

      		enthalpicBC_im=enthalpicBC_im+sum_factor[ijk]*N*u_k[ijk]*(chiBC*(phBs[ijk][1]*phCs[ijk][0]-phBs[ijk][0]*phCs[ijk][1]));

      		entropic=entropic-sum_factor[ijk]*((wAs[ijk][0]*phAs[ijk][0]+wAs[ijk][1]*phAs[ijk][1])+
             			                   	   (wBs[ijk][0]*phBs[ijk][0]+wBs[ijk][1]*phBs[ijk][1])+
             			                   	   (wCs[ijk][0]*phCs[ijk][0]+wCs[ijk][1]*phCs[ijk][1]));

      		entropic_im=entropic_im-sum_factor[ijk]*((wAs[ijk][1]*phAs[ijk][0]-wAs[ijk][0]*phAs[ijk][1])+
             			                   		   	 (wBs[ijk][1]*phBs[ijk][0]-wBs[ijk][0]*phBs[ijk][1])+
             			                   		   	 (wCs[ijk][1]*phCs[ijk][0]-wCs[ijk][0]*phCs[ijk][1]));
		}     

		if(world_rank == 0)
	    {
			MPI_Reduce(MPI_IN_PLACE, &enthalpicAB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(MPI_IN_PLACE, &enthalpicAC, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(MPI_IN_PLACE, &enthalpicBC, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(MPI_IN_PLACE, &enthalpicAB_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(MPI_IN_PLACE, &enthalpicAC_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(MPI_IN_PLACE, &enthalpicBC_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(MPI_IN_PLACE, &entropic, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(MPI_IN_PLACE, &entropic_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Reduce(&enthalpicAB, &enthalpicAB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(&enthalpicAC, &enthalpicAC, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(&enthalpicBC, &enthalpicBC, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(&enthalpicAB_im, &enthalpicAB_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(&enthalpicAC_im, &enthalpicAC_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(&enthalpicBC_im, &enthalpicBC_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(&entropic, &entropic, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		  	MPI_Reduce(&entropic_im, &entropic_im, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}

		enthalpic=enthalpicAB+enthalpicAC+enthalpicBC;
		enthalpic_im=enthalpicAB_im+enthalpicAC_im+enthalpicBC_im;

		entropic+=-log(parfQ);

    	freeOld=freeEnergy;
		freeEnergy=enthalpic+entropic;
	  	freeDiff=fabs(freeEnergy-freeOld);

	//10.Repeat until desired accuracy is reached//
	  	if(world_rank == 0)
		{
	    	if(iter==1||iter%10==0||iter>=maxIter)
	    	{
				FILE *fp=fopen("ing.dat","a");
				fprintf(fp, "%5d : %.8e, %.4e, err=%.4e\n", iter, freeEnergy, inCompMax, err);
				fprintf(fp, "Stress:\n s1: %.6e s2: %.6e %.6e s3: %.6e %.6e %.6e\n", stress[0], stress[3], stress[1], stress[5], stress[4], stress[2]);
				fprintf(fp, "P vectors:\n a1: %.6e a2: %.6e %.6e a3: %.6e %.6e %.6e\n\n\n", A_matrix[0], A_matrix[3], A_matrix[4], A_matrix[6], A_matrix[7], A_matrix[8]);
				fclose(fp);
	    	}

	    	end=MPI_Wtime();
	  		fp=fopen("time","a");
	    	fprintf(fp,"%d:\n",iter);
	    	fprintf(fp,"real=%.10lf\n",enthalpic+entropic); 
	    	fprintf(fp,"imaginary=%.10lf\n",enthalpic_im+entropic_im);
	    	fprintf(fp,"time per iteration = %.7E\n",end-start);
	    	fclose(fp);
	    }

	    // Find the maximum absolute stress //
  		stress_max=0.0;
  		for(i=0;i<nd;i++)
  		{
  			if(fabs(stress[i])>stress_max)stress_max=fabs(stress[i]);
  		}

	    MPI_Barrier(MPI_COMM_WORLD);

		MPI_Bcast(&freeDiff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}while(iter<maxIter&&(inCompMax>Sm1||freeDiff>Sm2||err>Sm3||stress_max>Sm4));

	if(world_rank == 0)
	{
		fp=fopen("ing.dat","a");
		fprintf(fp, "%5d : %.8e, %.4e, err=%.4e\n", iter, freeEnergy, inCompMax, err);
		fprintf(fp, "Stress:\n    %.6e  0.000000e+00  0.000000e+00\n    %.6e  %.6e  0.000000e+00\n    %.6e  %.6e  %.6e\n", stress[0], stress[3], stress[1], stress[5], stress[4], stress[2]);
		fprintf(fp, "A Matrix:\n    %.6e  %.6e  %.6e\n    %.6e  %.6e  %.6e\n    %.6e  %.6e  %.6e\n\n\n", A_matrix[0], A_matrix[1], A_matrix[2], A_matrix[3], A_matrix[4], A_matrix[5], A_matrix[6], A_matrix[7], A_matrix[8]);
		fclose(fp);
	}

	struct result_struct results;
	results.freeEnergy=freeEnergy;
	results.inCompMax=inCompMax;
	results.freeDiff=freeDiff;
	results.err=err;
	results.stress_max=stress_max;
	
	free(waDiff);
	free(wbDiff);
	free(wcDiff);
  
  	fftw_free(phAs);
  	fftw_free(phBs);
  	fftw_free(phCs);
  
	fftw_free(wAs);
	fftw_free(wBs);
	fftw_free(wCs);
  	free(wAnew);
	free(wBnew);
	free(wCnew);
  	fftw_free(wAnews);
  	fftw_free(wBnews);
  	fftw_free(wCnews);
  
	fftw_free(etas);

	free(del);
	free(outs);

	free(dudk);

	free(stress_FR_common);

	return results;
}


//*****Calculate forward and backward propagators*****//
//*****and single chain pertition function*****//
//*****and then calculate PhA & PhB*****//
double getConc(double *phA, double *phB, double *phC, double *wA, double *wB, double *wC)
{
	int i,j,k,iz;
	long ijk;
	double *qA,*qcA,*qB,*qcB,*qC,*qcC,*exp_wA,*exp_wB,*exp_wC,*dgdk_A,*dgdk_B,*dgdk_C;
	double *qJoint, *qBranch; 
	double parfQ, *qInt;
	double *g_k_A, *g_k_B, *g_k_C;
	double *stress_bond_common;
	fftw_complex *qAs,*qcAs,*qBs,*qcBs,*qCs,*qcCs;
	fftw_complex *qJoints;
	fftw_complex *qInts;
	double *g_k_n[nb_max-1];
	fftw_complex *qInts_n[nb_max-1];
	                 
	g_k_A=(double *)malloc(sizeof(double)*local_NaNbNc1);                 
	g_k_B=(double *)malloc(sizeof(double)*local_NaNbNc1);                 
	g_k_C=(double *)malloc(sizeof(double)*local_NaNbNc1);                 
	qA=(double *)malloc(sizeof(double)*local_Na_2_NbNc*NsA);
	qcA=(double *)malloc(sizeof(double)*local_Na_2_NbNc*NsA);
	qB=(double *)malloc(sizeof(double)*local_Na_2_NbNc*NsB);
	qcB=(double *)malloc(sizeof(double)*local_Na_2_NbNc*NsB);
	qC=(double *)malloc(sizeof(double)*local_Na_2_NbNc*NsC);
	qcC=(double *)malloc(sizeof(double)*local_Na_2_NbNc*NsC);
	qInt=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	exp_wA=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	exp_wB=(double *)malloc(sizeof(double)*local_Na_2_NbNc);
	exp_wC=(double *)malloc(sizeof(double)*local_Na_2_NbNc);

	qJoint=(double *)malloc(sizeof(double)*local_Na_2_NbNc*3);
	qJoints=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1*3);
	qBranch=(double *)malloc(sizeof(double)*local_Na_2_NbNc);

	stress_bond_common=(double *)malloc(sizeof(double)*local_NaNbNc1);

	qAs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1*NsA);
	qcAs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1*NsA);
	qBs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1*NsB);
	qcBs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1*NsB);
	qCs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1*NsC);
	qcCs=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1*NsC);
	qInts=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*local_NaNbNc1);

	dgdk_A=(double *)malloc(sizeof(double)*local_NaNbNc1);
	dgdk_B=(double *)malloc(sizeof(double)*local_NaNbNc1);
	dgdk_C=(double *)malloc(sizeof(double)*local_NaNbNc1);

	//Calculate all e^-w//
	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
  	{
  		exp_wA[ijk]=exp(-wA[ijk]/((double)(N)));
  	}
  	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=Na;i<Na+2;i++)
	{
		ijk=(k*Nb+j)*(Na+2)+i;
		exp_wA[ijk]=0.0;
	} 

  	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
  	{
  		exp_wB[ijk]=exp(-wB[ijk]/((double)(N)));
  	}
  	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=Na;i<Na+2;i++)
	{
		ijk=(k*Nb+j)*(Na+2)+i;
		exp_wB[ijk]=0.0;
	} 

	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
  	{
  		exp_wC[ijk]=exp(-wC[ijk]/((double)(N)));
  	}
  	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=Na;i<Na+2;i++)
	{
		ijk=(k*Nb+j)*(Na+2)+i;
		exp_wC[ijk]=0.0;
	} 

	//Define the transition probability//
	if(world_rank==0)
	{
		for(ijk=1;ijk<local_NaNbNc1;ijk++)
	  	{
	    	g_k_A[ijk]=((sin(k_norm[ijk]*blA))/(k_norm[ijk]*blA));				
	  	}
	  	g_k_A[0]=1.0;
	}
	else
	{
		for(ijk=0;ijk<local_NaNbNc1;ijk++)
	  	{
	    	g_k_A[ijk]=((sin(k_norm[ijk]*blA))/(k_norm[ijk]*blA));				
	  	}
	}

	if(world_rank==0)
	{
		for(ijk=1;ijk<local_NaNbNc1;ijk++)
	  	{
	    	g_k_B[ijk]=((sin(k_norm[ijk]*blB))/(k_norm[ijk]*blB));				
	  	}
	  	g_k_B[0]=1.0;
	}
	else
	{
		for(ijk=0;ijk<local_NaNbNc1;ijk++)
	  	{
	    	g_k_B[ijk]=((sin(k_norm[ijk]*blB))/(k_norm[ijk]*blB));				
	  	}
	}

	if(world_rank==0)
	{
		for(ijk=1;ijk<local_NaNbNc1;ijk++)
	  	{
	    	g_k_C[ijk]=((sin(k_norm[ijk]*blC))/(k_norm[ijk]*blC));				
	  	}
	  	g_k_C[0]=1.0;
	}
	else
	{
		for(ijk=0;ijk<local_NaNbNc1;ijk++)
	  	{
	    	g_k_C[ijk]=((sin(k_norm[ijk]*blC))/(k_norm[ijk]*blC));				
	  	}
	}

	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
  	{
  		qInt[ijk]=1.0;
  	}
  	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=Na;i<Na+2;i++)
	{
		ijk=(k*Nb+j)*(Na+2)+i;
		qInt[ijk]=0.0;
	} 

	//Calculation for all propagators initialized by q=1
	get_q(qA,qAs,exp_wA,g_k_A,qInt,NsA,1);     //1->NsA for qA

	get_q(qcB,qcBs,exp_wB,g_k_B,qInt,NsB,-1);    //NsB->1 for qcB

	get_q(qcC,qcCs,exp_wC,g_k_C,qInt,NsC,-1);    //NsC->1 for qcC

	//Calculation for the branching segment and get IC for the other propagators
	/*Calculation for qcA*/
	qInts_n[0]=qcBs;
	qInts_n[1]=qcCs;
	g_k_n[0]=g_k_B;
	g_k_n[1]=g_k_C;
	branching_propagation(3,qInt,qInts_n,g_k_n,g_k_A,exp_wA,qJoint,qJoints,0,0,0,qBranch);
	get_q(qcA,qcAs,exp_wA,g_k_A,qInt,NsA,-1);    //NsA to 1 for qcA

	/*Calculation for qB*/
	qInts_n[0]=qAs+(NsA-1)*local_NaNbNc1;
	qInts_n[1]=qcCs;
	g_k_n[0]=g_k_A;
	g_k_n[1]=g_k_C;
	branching_propagation(3,qInt,qInts_n,g_k_n,g_k_B,exp_wA,qJoint,qJoints,1,1,0,qBranch);
	get_q(qB,qBs,exp_wB,g_k_B,qInt,NsB,1);    //1 to NsB for qcA

	/*Calculation for qC*/
	qInts_n[1]=qcBs;
	g_k_n[1]=g_k_B;
	branching_propagation(3,qInt,qInts_n,g_k_n,g_k_C,exp_wA,qJoint,qJoints,2,0,0,qBranch);
	get_q(qC,qCs,exp_wC,g_k_C,qInt,NsC,1);    //1 to NsB for qcA

	//Calculate the single chain patition function//	
  	if(world_rank==0)
	{
		parfQ=qBs[(NsB-1)*local_NaNbNc1][0];
	}
	MPI_Bcast(&parfQ, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);








	



// //Test the single chain partition function of triblock copolymer//
// for(iz=0; iz<NsA; iz++)
// {
// 	parfQ=0.0;
// 	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
// 	{
// 		ijk=(k*Nb+j)*(Na+2)+i;
// 	   	parfQ+=(qA[iz*local_Na_2_NbNc+ijk]*qcA[iz*local_Na_2_NbNc+ijk]/exp_wA[ijk]);
// 	}
// 	MPI_Allreduce(MPI_IN_PLACE, &parfQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// 	parfQ/=NaNbNc;
// 	if(world_rank == 0)
// 	{
// 		printf("%.5e\n",parfQ);
// 	}
// }
// if(world_rank == 0)
// {
// 	printf("*********\n");
// }
// for(iz=0; iz<NsB; iz++)
// {
// 	parfQ=0.0;
// 	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
// 	{
// 		ijk=(k*Nb+j)*(Na+2)+i;
// 	   	parfQ+=(qB[iz*local_Na_2_NbNc+ijk]*qcB[iz*local_Na_2_NbNc+ijk]/exp_wB[ijk]);
// 	}
// 	MPI_Allreduce(MPI_IN_PLACE, &parfQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// 	parfQ/=NaNbNc;
// 	if(world_rank == 0)
// 	{
// 		printf("%.5e\n",parfQ);
// 	}
// }
// if(world_rank == 0)
// {
// 	printf("*********\n");
// }
// for(iz=0; iz<NsC; iz++)
// {
// 	parfQ=0.0;
// 	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
// 	{
// 		ijk=(k*Nb+j)*(Na+2)+i;
// 	   	parfQ+=(qC[iz*local_Na_2_NbNc+ijk]*qcC[iz*local_Na_2_NbNc+ijk]/exp_wC[ijk]);
// 	}
// 	MPI_Allreduce(MPI_IN_PLACE, &parfQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// 	parfQ/=NaNbNc;
// 	if(world_rank == 0)
// 	{
// 		printf("%.5e\n",parfQ);
// 	}
// }
// if(world_rank == 0)
// {
// 	printf("*********\n");
// }
// parfQ=0.0;
// for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
// {
// 	ijk=(k*Nb+j)*(Na+2)+i;
//    	parfQ+=(qBranch[ijk]*qJoint[ijk]/exp_wA[ijk]);
// }
// MPI_Allreduce(MPI_IN_PLACE, &parfQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// parfQ/=NaNbNc;
// if(world_rank == 0)
// {
// 	printf("%.5e\n",parfQ);
// 	printf("*********\n");
// }









  	//Calculate phA, phB and phC//
	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(k*Nb+j)*(Na+2)+i;

  		phA[ijk]=0.0;
  		phB[ijk]=0.0;
  		phC[ijk]=0.0;
  
		for(iz=0;iz<NsA;iz++)
		{
			phA[ijk]+=(qA[iz*local_Na_2_NbNc+ijk]*qcA[iz*local_Na_2_NbNc+ijk]);
		}
		phA[ijk]+=qBranch[ijk]*qJoint[ijk];

		for(iz=0;iz<NsB;iz++)
		{
			phB[ijk]+=(qB[iz*local_Na_2_NbNc+ijk]*qcB[iz*local_Na_2_NbNc+ijk]);
		}

		for(iz=0;iz<NsC;iz++)
		{
			phC[ijk]+=(qC[iz*local_Na_2_NbNc+ijk]*qcC[iz*local_Na_2_NbNc+ijk]);
		}
    	phA[ijk]/=parfQ*N*exp_wA[ijk];
    	phB[ijk]/=parfQ*N*exp_wB[ijk];
    	phC[ijk]/=parfQ*N*exp_wC[ijk];
	}

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=Na;i<Na+2;i++)
	{
		ijk=(k*Nb+j)*(Na+2)+i;

  		phA[ijk]=0.0;
  		phB[ijk]=0.0;
  		phC[ijk]=0.0;
  	}
  









// //Test the calculation of phi//
// double average_A=0.0, average_B=0.0, average_C=0.0;
// for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
// {
// 	average_A+=phA[ijk];
// 	average_B+=phB[ijk];
// 	average_C+=phC[ijk];
// }
// MPI_Allreduce(MPI_IN_PLACE, &average_A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// MPI_Allreduce(MPI_IN_PLACE, &average_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// MPI_Allreduce(MPI_IN_PLACE, &average_C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// average_A/=NaNbNc;
// average_B/=NaNbNc;
// average_C/=NaNbNc;
// if(world_rank==0)
// {
// 	printf("average = %lf\t%lf\t%lf\n", average_A, average_B, average_C);
// }










	//Stress//	
	if(world_rank==0)
	{
		for(ijk=1; ijk<local_NaNbNc1; ijk++)
		{
			dgdk_A[ijk]=(blA/(2*k_norm[ijk]))*(k_norm[ijk]*blA*cos(k_norm[ijk]*blA)-sin(k_norm[ijk]*blA))/(k_norm[ijk]*k_norm[ijk]*blA*blA);
		}
		dgdk_A[0]=0.0;
	}
	else
	{
		for(ijk=0; ijk<local_NaNbNc1; ijk++)
		{
			dgdk_A[ijk]=(blA/(2*k_norm[ijk]))*(k_norm[ijk]*blA*cos(k_norm[ijk]*blA)-sin(k_norm[ijk]*blA))/(k_norm[ijk]*k_norm[ijk]*blA*blA);
		}
	}	
	
	if(world_rank==0)
	{
		for(ijk=1; ijk<local_NaNbNc1; ijk++)
		{
			dgdk_B[ijk]=(blB/(2*k_norm[ijk]))*(k_norm[ijk]*blB*cos(k_norm[ijk]*blB)-sin(k_norm[ijk]*blB))/(k_norm[ijk]*k_norm[ijk]*blB*blB);
		} 
		dgdk_B[0]=0.0;
	}
	else
	{
		for(ijk=0; ijk<local_NaNbNc1; ijk++)
		{
			dgdk_B[ijk]=(blB/(2*k_norm[ijk]))*(k_norm[ijk]*blB*cos(k_norm[ijk]*blB)-sin(k_norm[ijk]*blB))/(k_norm[ijk]*k_norm[ijk]*blB*blB);
		} 
	}
	
	if(world_rank==0)
	{
		for(ijk=1; ijk<local_NaNbNc1; ijk++)
		{
			dgdk_C[ijk]=(blC/(2*k_norm[ijk]))*(k_norm[ijk]*blC*cos(k_norm[ijk]*blC)-sin(k_norm[ijk]*blC))/(k_norm[ijk]*k_norm[ijk]*blC*blC);
		} 
		dgdk_C[0]=0.0;
	}
	else
	{
		for(ijk=0; ijk<local_NaNbNc1; ijk++)
		{
			dgdk_C[ijk]=(blC/(2*k_norm[ijk]))*(k_norm[ijk]*blC*cos(k_norm[ijk]*blC)-sin(k_norm[ijk]*blC))/(k_norm[ijk]*k_norm[ijk]*blC*blC);
		} 
	}

	for(ijk=0; ijk<local_NaNbNc1; ijk++)
	{
		stress_bond_common[ijk]=0.0;

		for(iz=0;iz<NsA-1;iz++)
		{
			stress_bond_common[ijk]+=(1.0/parfQ)*dgdk_A[ijk]*(qAs[iz*local_NaNbNc1+ijk][0]*qcAs[(iz+1)*local_NaNbNc1+ijk][0]+qAs[iz*local_NaNbNc1+ijk][1]*qcAs[(iz+1)*local_NaNbNc1+ijk][1]);
		}
		stress_bond_common[ijk]+=(1.0/parfQ)*dgdk_A[ijk]*(qAs[(NsA-1)*local_NaNbNc1+ijk][0]*qJoints[ijk][0]+qAs[(NsA-1)*local_NaNbNc1+ijk][1]*qJoints[ijk][1]);
		for(iz=0;iz<NsB-1;iz++)
		{
			stress_bond_common[ijk]+=(1.0/parfQ)*dgdk_B[ijk]*(qBs[iz*local_NaNbNc1+ijk][0]*qcBs[(iz+1)*local_NaNbNc1+ijk][0]+qBs[iz*local_NaNbNc1+ijk][1]*qcBs[(iz+1)*local_NaNbNc1+ijk][1]);
		}
		stress_bond_common[ijk]+=(1.0/parfQ)*dgdk_B[ijk]*(qcBs[ijk][0]*qJoints[local_NaNbNc1+ijk][0]+qcBs[ijk][1]*qJoints[local_NaNbNc1+ijk][1]);

		for(iz=0;iz<NsC-1;iz++)
		{
			stress_bond_common[ijk]+=(1.0/parfQ)*dgdk_C[ijk]*(qCs[iz*local_NaNbNc1+ijk][0]*qcCs[(iz+1)*local_NaNbNc1+ijk][0]+qCs[iz*local_NaNbNc1+ijk][1]*qcCs[(iz+1)*local_NaNbNc1+ijk][1]);
		}
		stress_bond_common[ijk]+=(1.0/parfQ)*dgdk_C[ijk]*(qcCs[ijk][0]*qJoints[2*local_NaNbNc1+ijk][0]+qcCs[ijk][1]*qJoints[2*local_NaNbNc1+ijk][1]);
	} 

	for(i=0;i<nd;i++)
	{
		stress_bond[i]=0.0;
	}
	for(ijk=0; ijk<local_NaNbNc1; ijk++)
	{
		stress_bond[0]+=sum_factor[ijk]*dk2_da[ijk]*stress_bond_common[ijk];
		stress_bond[1]+=sum_factor[ijk]*dk2_db[ijk]*stress_bond_common[ijk];
		stress_bond[2]+=sum_factor[ijk]*dk2_dc[ijk]*stress_bond_common[ijk];
		stress_bond[3]+=sum_factor[ijk]*dk2_dd[ijk]*stress_bond_common[ijk];
		stress_bond[4]+=sum_factor[ijk]*dk2_de[ijk]*stress_bond_common[ijk];
		stress_bond[5]+=sum_factor[ijk]*dk2_df[ijk]*stress_bond_common[ijk];
	}  

	free(g_k_A);
	free(g_k_B);
	free(g_k_C);
	free(qA);
	free(qB);
	free(qC);
	free(qcA);
	free(qcB);
	free(qcC);
	free(qInt);
	free(exp_wA);
	free(exp_wB);
	free(exp_wC);
	free(stress_bond_common);

	free(qJoint);
	free(qBranch);
	fftw_free(qJoints);

	fftw_free(qAs);
	fftw_free(qcAs);
	fftw_free(qBs);
	fftw_free(qcBs);
	fftw_free(qCs);
	fftw_free(qcCs);
	fftw_free(qInts);

	free(dgdk_A);
	free(dgdk_B);
	free(dgdk_C);

	return parfQ;
}


//*****Calculate propagator*****//
void get_q(double *prop, fftw_complex *props, double *exp_w, double *g_k, double *qInt, int ns, int sign)
{
  	int i,j,k,iz;
  	long ijk;
  		  
	if(sign==1)
	{ 
		// Single-step multiplication for the first segment and get its fourier coefficients
		for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
		{
			prop[ijk]=qInt[ijk]*exp_w[ijk];
		} 

		forward_fft_mpi(prop, out);

		for(ijk=0;ijk<local_NaNbNc1;ijk++)
		{
			props[ijk][0]=out[ijk][0];		
			props[ijk][1]=out[ijk][1];		
		}

		// Propagation for the rest of the segments
		for(iz=1;iz<ns;iz++)
		{     		
			for(ijk=0;ijk<local_NaNbNc1;ijk++)
			{
        		out[ijk][0]*=g_k[ijk];		
        		out[ijk][1]*=g_k[ijk];	
     		}

     		prop+=local_Na_2_NbNc;

			backward_fft_mpi(out, prop);
	
  			for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
      		{
        		prop[ijk]*=exp_w[ijk];
      		}     		

      		forward_fft_mpi(prop, out);	

      		props+=local_NaNbNc1;
			
			for(ijk=0;ijk<local_NaNbNc1;ijk++)
			{
        		props[ijk][0]=out[ijk][0];		
        		props[ijk][1]=out[ijk][1];		
     		}
		}

		prop-=local_Na_2_NbNc*(ns-1);
		props-=local_NaNbNc1*(ns-1);
	}
	else 
	{
		prop+=local_Na_2_NbNc*(ns-1);
		props+=local_NaNbNc1*(ns-1);

		// Single-step multiplication for the first segment and get its fourier coefficients
		for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
		{
			prop[ijk]=qInt[ijk]*exp_w[ijk];
		}

		forward_fft_mpi(prop, out);

		for(ijk=0;ijk<local_NaNbNc1;ijk++)
		{
			props[ijk][0]=out[ijk][0];		
			props[ijk][1]=out[ijk][1];		
		}
		
		// Propagation for the rest of the segments
		for(iz=1;iz<ns;iz++)
		{				
	    	for(ijk=0;ijk<local_NaNbNc1;ijk++)
			{
        		out[ijk][0]*=g_k[ijk];	
        		out[ijk][1]*=g_k[ijk];	
      		}

      		prop-=local_Na_2_NbNc;

			backward_fft_mpi(out, prop);
	
      		for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
     		{
        		prop[ijk]*=exp_w[ijk];
      		}

      		forward_fft_mpi(prop, out);

      		props-=local_NaNbNc1;
	
			for(ijk=0;ijk<local_NaNbNc1;ijk++)
			{
        		props[ijk][0]=out[ijk][0];		
        		props[ijk][1]=out[ijk][1];		
     		}
    	}
 	}
}


void conjunction_propagation(double *qInt, fftw_complex *qInts, double *g_k)
{
	int i,j,k;
	long ijk;
	
	for(ijk=0;ijk<local_NaNbNc1;ijk++)
	{
		out[ijk][0]=qInts[ijk][0]*g_k[ijk];		
		out[ijk][1]=qInts[ijk][1]*g_k[ijk];		
	}

	backward_fft_mpi(out, qInt);
}


void branching_propagation(int num_branches, double *qInt, fftw_complex *qInts_n[], double *g_k_n[], double *g_k, double *exp_w, double *qJoint, fftw_complex *qJoints, int idxJoint, int IF_STORE, int idxStore, double *qBranch)
{
	int i,j,k,bn;
	long ijk;
	double *tempReal;

	tempReal=(double *)malloc(sizeof(double)*local_Na_2_NbNc);

	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
  	{
		qInt[ijk]=1.0;
	}

	for(bn=0;bn<num_branches-1;bn++)
	{
		conjunction_propagation(tempReal, qInts_n[bn], g_k_n[bn]);

		for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
	  	{
			qInt[ijk]*=tempReal[ijk];
		}

		if(IF_STORE)
		{
			if(idxStore==bn)
			{
				for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
	  			{
					qBranch[ijk]=tempReal[ijk]*exp_w[ijk];
				}
			}
		}
	}

	qJoint+=idxJoint*local_Na_2_NbNc;
	qJoints+=idxJoint*local_NaNbNc1;

	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
  	{
		qJoint[ijk]=qInt[ijk]*exp_w[ijk];
	}

	forward_fft_mpi(qJoint, out);	
			
	for(ijk=0;ijk<local_NaNbNc1;ijk++)
	{
		qJoints[ijk][0]=out[ijk][0];		
		qJoints[ijk][1]=out[ijk][1];		
	}

	for(ijk=0;ijk<local_NaNbNc1;ijk++)
	{
		out[ijk][0]*=g_k[ijk];		
		out[ijk][1]*=g_k[ijk];	
	}

	backward_fft_mpi(out, qInt);

	qJoint-=idxJoint*local_Na_2_NbNc;
	qJoints-=idxJoint*local_NaNbNc1;

	free(tempReal);
}


// Wrapper of fftw with forward transform normalized
void forward_fft_mpi(double *f_r, fftw_complex *f_k)
{
	long ijk;

	fftw_mpi_execute_dft_r2c(p_forward, f_r, f_k);

	for(ijk=0;ijk<local_NaNbNc1;ijk++)
	{
		f_k[ijk][0]/=NaNbNc;		
		f_k[ijk][1]/=NaNbNc;		
	}
}


void backward_fft_mpi(fftw_complex *f_k, double *f_r)
{
	fftw_mpi_execute_dft_c2r(p_backward, f_k, f_r);
}

  
//Calculate the error between in and out w field//          
double error_cal(double *waDiffs, double *wbDiffs, double *wcDiffs, double *wAs, double *wBs, double *wCs)
{
	double err_dif, err_w, err;
	int ijk;

	err = 0.0;
	err_dif = 0.0;
	err_w = 0.0;

	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
  	{
		err_dif += pow(*(waDiffs++),2)+pow(*(wbDiffs++),2)+pow(*(wcDiffs++),2);
		err_w += pow(*(wAs++),2)+pow(*(wBs++),2)+pow(*(wCs++),2);
	}
	wAs-=local_Na_2_NbNc;
	wBs-=local_Na_2_NbNc;
	wCs-=local_Na_2_NbNc;
	waDiffs-=local_Na_2_NbNc;
	wbDiffs-=local_Na_2_NbNc;
	wcDiffs-=local_Na_2_NbNc;
	
	if(world_rank == 0)
    {
		MPI_Reduce(MPI_IN_PLACE, &err_dif, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	  	MPI_Reduce(MPI_IN_PLACE, &err_w, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Reduce(&err_dif, &err_dif, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	  	MPI_Reduce(&err_w, &err_w, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	err = err_dif/err_w;
	err = sqrt(err);

	return err;
}


void cal_B_matrix(double *B_matrix)
{
	int i,s;
	gsl_matrix_view xv, yv;        // gsl_matrix(vector)_view is used to create a view of an array by function gsl_matrix(vector)_view_array
	gsl_permutation *p;
	double *A_matrix_copy;

	A_matrix_copy = (double *)calloc(9,sizeof(double));

	for(i=0;i<9;i++)
  	{
  		A_matrix_copy[i]=A_matrix[i];
  	}

  	xv = gsl_matrix_view_array(A_matrix_copy, 3, 3);
  	yv = gsl_matrix_view_array(B_matrix, 3, 3);

  	p = gsl_permutation_alloc(3);

  	gsl_linalg_LU_decomp(&xv.matrix, p, &s);             // After the LU decomposition the original matrix will be changed to store the elements of L & U matrices
  	gsl_linalg_LU_invert(&xv.matrix, p, &yv.matrix);     // Take the inversion of matrix x according to its LU decomposition matrices and store the results in matrix y

  	gsl_permutation_free(p);

  	for(i=0;i<9;i++)
  	{
  		B_matrix[i]*=2.0*Pi;
  	}

  	free(A_matrix_copy);
}


void cal_k_vectors()
{
	int i,j,k;
	long ijk;
	double *B_matrix;
	double b1[3], b2[3], b3[3];

	B_matrix=(double *)calloc(9,sizeof(double));

	cal_B_matrix(B_matrix);

  	for(i=0;i<3;i++)
  	{
  		b1[i]=B_matrix[3*i];
  	}
  	for(i=0;i<3;i++)
  	{
  		b2[i]=B_matrix[3*i+1];
  	}
  	for(i=0;i<3;i++)
  	{
  		b3[i]=B_matrix[3*i+2];
  	}

	double b1_sq, b2_sq, b3_sq, b1b2, b1b3, b2b3;
	b1_sq=b1[0]*b1[0]+b1[1]*b1[1]+b1[2]*b1[2];
	b2_sq=b2[0]*b2[0]+b2[1]*b2[1]+b2[2]*b2[2];
	b3_sq=b3[0]*b3[0]+b3[1]*b3[1]+b3[2]*b3[2];
	b1b2=b1[0]*b2[0]+b1[1]*b2[1]+b1[2]*b2[2];
	b1b3=b1[0]*b3[0]+b1[1]*b3[1]+b1[2]*b3[2];
	b2b3=b2[0]*b3[0]+b2[1]*b3[1]+b2[2]*b3[2];

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);
		k_square[ijk]=h_idx[i]*h_idx[i]*b1_sq + k_idx[j]*k_idx[j]*b2_sq + l_idx[k+local_Nc_start]*l_idx[k+local_Nc_start]*b3_sq + 2.0*h_idx[i]*k_idx[j]*b1b2 + 2.0*h_idx[i]*l_idx[k+local_Nc_start]*b1b3 + 2.0*k_idx[j]*l_idx[k+local_Nc_start]*b2b3;
		k_norm[ijk]=sqrt(k_square[ijk]);
	}

	free(B_matrix);
}


void cal_der_k2()
{
	int i,j,k;
	long ijk;
	double *hh, *kk, *ll, *hk, *kl, *hl;
	double a,b,c,d,e,f;
	double pref, hh_term, kk_term, ll_term, hk_term, kl_term, hl_term;
	double volume;
	
	hh=(double *)malloc(sizeof(double)*local_NaNbNc1);
	kk=(double *)malloc(sizeof(double)*local_NaNbNc1);
	ll=(double *)malloc(sizeof(double)*local_NaNbNc1);
	hk=(double *)malloc(sizeof(double)*local_NaNbNc1);
	kl=(double *)malloc(sizeof(double)*local_NaNbNc1);
	hl=(double *)malloc(sizeof(double)*local_NaNbNc1);

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);

		hh[ijk]=h_idx[i]*h_idx[i];
		kk[ijk]=k_idx[j]*k_idx[j];
		ll[ijk]=l_idx[k+local_Nc_start]*l_idx[k+local_Nc_start];
		hk[ijk]=h_idx[i]*k_idx[j];
		kl[ijk]=k_idx[j]*l_idx[k+local_Nc_start];
		hl[ijk]=h_idx[i]*l_idx[k+local_Nc_start];
	}

	a=A_matrix[0];
	b=A_matrix[4];
	c=A_matrix[8];
	d=A_matrix[3];
	e=A_matrix[7];
	f=A_matrix[6];

	volume=a*b*c;
	pref=-8.0*Pi*Pi/(volume*volume);

	double c2_e2=c*c+e*e;

	hh_term=b*b*(c*c+f*f)+d*d*c2_e2-2.0*b*d*e*f;
	hk_term=-a*d*c2_e2+a*b*e*f;
	hl_term=a*b*(d*e-b*f);

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);

		dk2_da[ijk]=pref*(1.0/a)*(hh[ijk]*hh_term+hk[ijk]*hk_term+hl[ijk]*hl_term);
	}

	hh_term=d*d*c2_e2-b*d*e*f;
	kk_term=a*a*c2_e2;
	hk_term=-2.0*a*d*c2_e2+a*b*e*f;
	kl_term=-a*a*b*e;
	hl_term=a*b*d*e;

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);

		dk2_db[ijk]=pref*(1.0/b)*(hh[ijk]*hh_term+kk[ijk]*kk_term+hk[ijk]*hk_term+kl[ijk]*kl_term+hl[ijk]*hl_term);
	}

	hh_term=d*d*e*e+b*b*f*f-2.0*d*e*b*f;
	kk_term=a*a*e*e;
	ll_term=a*a*b*b;
	hk_term=-2.0*a*(d*e*e-b*e*f);
	kl_term=-2.0*a*a*b*e;
	hl_term=2.0*a*b*(d*e-b*f);

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);

		dk2_dc[ijk]=pref*(1.0/c)*(hh[ijk]*hh_term+kk[ijk]*kk_term+ll[ijk]*ll_term+hk[ijk]*hk_term+kl[ijk]*kl_term+hl[ijk]*hl_term);
	}

	hh_term=-d*c2_e2+e*b*f;
	hk_term=a*c2_e2;
	hl_term=-a*b*e;

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);

		dk2_dd[ijk]=pref*(hh[ijk]*hh_term+hk[ijk]*hk_term+hl[ijk]*hl_term);
	}

	hh_term=-d*(d*e-b*f);
	kk_term=-a*a*e;
	hk_term=a*(2.0*d*e-b*f);
	kl_term=a*a*b;
	hl_term=-a*b*d;

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);

		dk2_de[ijk]=pref*(hh[ijk]*hh_term+kk[ijk]*kk_term+hk[ijk]*hk_term+kl[ijk]*kl_term+hl[ijk]*hl_term);
	}

	hh_term=-b*(b*f-d*e);
	hk_term=-a*b*e;
	hl_term=a*b*b;

	for(k=0;k<local_Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Nah1;i++)
	{
		ijk=(long)((k*Nb+j)*Nah1+i);

		dk2_df[ijk]=pref*(hh[ijk]*hh_term+hk[ijk]*hk_term+hl[ijk]*hl_term);
	}

	free(hh);
	free(kk);
	free(ll);
	free(hk);
	free(kl);
	free(hl);
}


//Update the record of data of preceding N_hist iterations used in Anderson mixing//
void update_flds_hist(double *waDiff, double *wbDiff, double *wcDiff, double *wAnew, double *wBnew, double *wCnew, double *del, double *outs)
{
	int ijk, j;

	del+=hist_Na*(N_hist-1);
	outs+=hist_Na*(N_hist-1);

	for(j=0;j<N_hist-1;j++)
	{
		
		for(ijk=0;ijk<hist_Na;ijk++)
		{
			del[ijk] = del[ijk-hist_Na];
			outs[ijk] = outs[ijk-hist_Na];
		}
		del-=hist_Na;
		outs-=hist_Na;
	}
	
	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
	{
		*(del++) = *(waDiff++);
		*(outs++) = *(wAnew++);
	}
	
	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
	{
		*(del++) = *(wbDiff++);
		*(outs++) = *(wBnew++);
	}
	
	for(ijk=0;ijk<local_Na_2_NbNc;ijk++)
	{
		*(del++) = *(wcDiff++);
		*(outs++) = *(wCnew++);
	}

	for(j=0;j<nd;j++)
	{
		del[j] = stress[j]*sqrt(A_matrix[0]*A_matrix[4]*A_matrix[8]);
		outs[j] = lattice_para_new[j];
	}
}

/*********************************************************************/
/*
  Anderson mixing [O(Na)]

  CHECKED
*/

void Anderson_mixing(double *del, double *outs, int N_rec, double *wA, double *wB, double *wC)
{
	int i, k, ijk;
	int n, m;
	double *U, *V, *A, temp; 
	int s;

	gsl_matrix_view uGnu;
	gsl_vector_view vGnu, aGnu;
	gsl_permutation *p;

	U = (double *)malloc(sizeof(double)*(N_rec-1)*(N_rec-1));
	V = (double *)malloc(sizeof(double)*(N_rec-1));
	A = (double *)malloc(sizeof(double)*(N_rec-1));
  
  	/* 
	Calculate the U-matrix and the V-vector 
	Follow Shuang, and add the A and B components together.
  	*/
      
	for(n=0; n<N_rec-1; n++)
	{
		temp=0.0;
      	
		for(ijk=0; ijk<hist_Na; ijk++)
		{ 
	  		temp += ( del[ijk] - del[(n+1)*hist_Na+ijk] ) * del[ijk];
		}
		if(world_rank == 0)
		{
			MPI_Reduce(MPI_IN_PLACE, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Reduce(&temp, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}
		
		V[n] = temp;
      
		for(m=n; m<N_rec-1; m++)
		{
			temp = 0.0;

	  		for (ijk=0; ijk<hist_Na; ijk++)
			{
	    		temp += ( del[ijk] - del[(n+1)*hist_Na+ijk] ) * ( del[ijk] - del[(m+1)*hist_Na+ijk] );
			}
			if(world_rank == 0)
			{
				MPI_Reduce(MPI_IN_PLACE, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			}
			else
			{
				MPI_Reduce(&temp, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			}
			U[(N_rec-1)*n+m] = temp;
	  		U[(N_rec-1)*m+n] = U[(N_rec-1)*n+m];
		}
	}
  
	/* Calculate A - uses GNU LU decomposition for U A = V */
  	if(world_rank == 0)
  	{
  		uGnu = gsl_matrix_view_array(U, N_rec-1, N_rec-1);
		vGnu = gsl_vector_view_array(V, N_rec-1);
		aGnu = gsl_vector_view_array(A, N_rec-1);

		p = gsl_permutation_alloc(N_rec-1);

		gsl_linalg_LU_decomp(&uGnu.matrix, p, &s);
	  
		gsl_linalg_LU_solve(&uGnu.matrix, p, &vGnu.vector, &aGnu.vector);

		gsl_permutation_free(p);
  	}
	
	MPI_Bcast(A, N_rec-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* Update omega */
	
	for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
	{
		*(wA++) = *(outs++);
	}
	wA-=local_Na_2_NbNc;

	for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
	{
		*(wB++) = *(outs++);
	}
	wB-=local_Na_2_NbNc;

	for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
	{
		*(wC++) = *(outs++);
	}
	wC-=local_Na_2_NbNc;

	for(i=0;i<nd;i++)
	{
		lattice_para[i] = outs[i];
	}

	outs-=local_Na_2_NbNc*3;

	for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
	{
		for(n=0; n<N_rec-1; n++)
		{
			wA[ijk] += A[n]*(outs[(n+1)*hist_Na+ijk] - outs[ijk]);
		}
	}

	outs+=local_Na_2_NbNc;

	for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
	{
		for(n=0; n<N_rec-1; n++)
		{
			wB[ijk] += A[n]*(outs[(n+1)*hist_Na+ijk] - outs[ijk]);
		}
	}

	outs+=local_Na_2_NbNc;

	for(ijk=0; ijk<local_Na_2_NbNc; ijk++)
	{
		for(n=0; n<N_rec-1; n++)
		{
			wC[ijk] += A[n]*(outs[(n+1)*hist_Na+ijk] - outs[ijk]);
		}
	}

	outs+=local_Na_2_NbNc;

	for(i=0;i<nd;i++)
	{
		for(n=0; n<N_rec-1; n++)
		{
			lattice_para[i] += A[n]*(outs[(n+1)*hist_Na+i] - outs[i]);
		}
	}
	
	free(A);
	free(V);
	free(U);
}


//Initialization//
void init_Reading(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC)
{
  	int i,j,k;
	long ijk;
  	FILE *fp;
  	fp=fopen("phiin.dat","r");
  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=0;i<Na;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	fscanf(fp,"%lf %lf %lf %lf %lf %lf\n",&phA[ijk],&phB[ijk],&phC[ijk],&wA[ijk],&wB[ijk],&wC[ijk]);
			}
		}
	}		
  	fclose(fp); 

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	phA[ijk]=0.0;
			  	phB[ijk]=0.0;
			  	phC[ijk]=0.0;
			  	wA[ijk]=0.0;
			  	wB[ijk]=0.0;
			  	wC[ijk]=0.0;
			}
		}
	}
}


void init_Random_primitive(double *phi_min, double *phi_maj)
{
	int i,j,k;
	long ijk;
	double rx, ry, rz;
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	FILE *fp=fopen("phi_min.dat","w");

	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		phi_min[ijk]=drand48();
		phi_maj[ijk]=1.0-phi_min[ijk];

		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]); 
	}
	fclose(fp);

	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_Disordered_primitive(double *phi_min, double *phi_maj)
{
  	int i,j,k;
	long ijk;
	double rx,ry,rz;
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);
		
		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		phi_min[ijk]=0.5;
  		phi_maj[ijk]=0.5;

  		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
  	}
  	fclose(fp);

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_Lamella_primitive(double *phi_min, double *phi_maj)
{
  	int i,j,k;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");

	r0sqd=pow(A_matrix[0]*0.5/2.0,2.0); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 
		rx0=0.5*A_matrix[0];
		rsqd=(rx-rx0)*(rx-rx0);
		if(rsqd<=r0sqd)    
	  	{
		  	phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}
	  	else
		{			
			phi_min[ijk]=1.0;
      		phi_maj[ijk]=0.0;			
		}
 		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
  	}
  	fclose(fp);

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);	

			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_HEX_primitive(double *phi_min, double *phi_maj)
{
	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[4][2]={{0,0},
			           {0,1},
			           {1,0},
			           {1,1}};
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
  
	r0sqd=pow(0.375*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		counter=0;
		for(m=0;m<4;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

  		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
    } 
    fclose(fp);

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	} 
}


void init_BCC_primitive(double *phi_min, double *phi_maj)
{
	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[8][3]={{0, 0, 0},
					   {1, 0, 0},
					   {0, 1, 0},
					   {1, 1, 0},
					   {0, 0, 1},
					   {1, 0, 1},
					   {0, 1, 1},
					   {1, 1, 1}};
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
  
	r0sqd=pow(0.375*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		counter=0;
		for(m=0;m<8;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

  		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
    } 
    fclose(fp);

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	} 
}


void init_FCC_primitive(double *phi_min, double *phi_maj)
{
	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[8][3]={{0, 0, 0},
					   {1, 0, 0},
					   {0, 1, 0},
					   {1, 1, 0},
					   {0, 0, 1},
					   {1, 0, 1},
					   {0, 1, 1},
					   {1, 1, 1}};
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
  
	r0sqd=pow(0.375*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		counter=0;
		for(m=0;m<8;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

  		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
    } 
    fclose(fp);

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	} 
}


void init_HCP_primitive(double *phi_min, double *phi_maj)
{
	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[2][3]={{0.66667, 0.33333, 0.75000},
					   {0.33333, 0.66667, 0.25000}};
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
  
	r0sqd=pow(0.05*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		counter=0;
		for(m=0;m<2;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

  		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
    } 
    fclose(fp);

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	} 
}


void init_C14_primitive(double *phi_min, double *phi_maj)
{
	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[22][3]={{0.00000, 1.00000, 1.00000},
					    {0.00000, 1.00000, 0.50000},
					    {0.00000, 1.00000, 0.00000},
					    {1.00000, 1.00000, 1.00000},
					    {1.00000, 1.00000, 0.50000},
					    {1.00000, 1.00000, 0.00000},
					    {0.00000, 0.00000, 1.00000},
					    {0.00000, 0.00000, 0.50000},
					    {0.00000, 0.00000, 0.00000},
					    {1.00000, 0.00000, 1.00000},
					    {1.00000, 0.00000, 0.50000},
					    {1.00000, 0.00000, 0.00000},
					    {0.33333, 0.66667, 0.43800},
					    {0.33333, 0.66667, 0.06200},
					    {0.66667, 0.33333, 0.93800},
					    {0.66667, 0.33333, 0.56200},
					    {0.66100, 0.83050, 0.75000},
					    {0.16950, 0.83050, 0.75000},
					    {0.16950, 0.33900, 0.75000},
					    {0.83050, 0.66100, 0.25000},
					    {0.83050, 0.16950, 0.25000},
					    {0.33900, 0.16950, 0.25000}};
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
  
	r0sqd=pow(0.02*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		counter=0;
		for(m=0;m<22;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

  		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
    } 
    fclose(fp);

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	} 
}


void init_C15_primitive(double *phi_min, double *phi_maj)
{
  	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[34][3]={{0.87500, 0.37500, 0.12500},
						{0.87500, 0.12500, 0.37500},
						{0.87500, 0.87500, 0.62500},
						{0.87500, 0.62500, 0.87500},
						{0.62500, 0.87500, 0.87500},
						{0.62500, 0.62500, 0.62500},
						{0.62500, 0.37500, 0.37500},
						{0.62500, 0.12500, 0.12500},
						{0.37500, 0.87500, 0.12500},
						{0.37500, 0.62500, 0.37500},
						{0.37500, 0.37500, 0.62500},
						{0.37500, 0.12500, 0.87500},
						{0.12500, 0.37500, 0.87500},
						{0.12500, 0.12500, 0.62500},
						{0.12500, 0.87500, 0.37500},
						{0.12500, 0.62500, 0.12500},
						{1.00000, 0.00000, 0.00000},
						{1.00000, 1.00000, 0.00000},
						{1.00000, 0.50000, 0.50000},
						{1.00000, 0.00000, 1.00000},
						{1.00000, 1.00000, 1.00000},
						{0.75000, 0.75000, 0.25000},
						{0.75000, 0.25000, 0.75000},
						{0.50000, 0.50000, 0.00000},
						{0.50000, 0.00000, 0.50000},
						{0.50000, 1.00000, 0.50000},
						{0.50000, 0.50000, 1.00000},
						{0.25000, 0.25000, 0.25000},
						{0.25000, 0.75000, 0.75000},
						{0.00000, 0.00000, 0.00000},
						{0.00000, 1.00000, 0.00000},
						{0.00000, 0.50000, 0.50000},
						{0.00000, 0.00000, 1.00000},
						{0.00000, 1.00000, 1.00000}};
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
 
	r0sqd=pow(0.03*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2];  

		counter=0;
		for(m=0;m<34;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

      	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);      
    } 
    fclose(fp); 

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}        
}


void init_A15_primitive(double *phi_min, double *phi_maj)
{
  	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[21][3]={{0, 0, 0},
 						{1, 0, 0},
 						{0, 1, 0},
 						{0, 0, 1},
 						{1, 1, 0},
 						{1, 0, 1},
 						{0, 1, 1},
 						{1, 1, 1},
 						{0.5, 0.5, 0.5},
 						{0.75000, 1.00000, 0.50000},
 						{0.25000, 1.00000, 0.50000},
 						{0.50000, 0.75000, 0.00000},
 						{0.50000, 0.25000, 0.00000},
 						{1.00000, 0.50000, 0.25000},
 						{0.00000, 0.50000, 0.25000},
 						{1.00000, 0.50000, 0.75000},
 						{0.00000, 0.50000, 0.75000},
 						{0.50000, 0.75000, 1.00000},
 						{0.50000, 0.25000, 1.00000},
 						{0.75000, 0.00000, 0.50000},
 						{0.25000, 0.00000, 0.50000}};
 	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");

	r0sqd=pow(0.09*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2];  

		counter=0;
		for(m=0;m<21;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

      	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);      
    } 
    fclose(fp);   

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}      
}


void init_Sigma_primitive(double *phi_min, double *phi_maj)
{
  	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[47][3]={{0, 0, 0},
 						{1, 0, 1},
 						{0, 1, 0},
 						{0, 0, 1},
 						{1, 1, 0},
 						{1, 0, 0},
 						{0, 1, 1},
 						{1, 1, 1},
 						{0.5, 0.5, 0.5},
 						{0.06530, 0.73760, 0.00000},
 						{0.06530, 0.73760, 1.00000},
 						{0.26240, 0.93470, 0.00000},
 						{0.26240, 0.93470, 1.00000},
 						{0.56530, 0.76240, 0.50000},
 						{0.23760, 0.43470, 0.50000},
 						{0.76240, 0.56530, 0.50000},
 						{0.43470, 0.23760, 0.50000},
 						{0.73760, 0.06530, 0.00000},
 						{0.73760, 0.06530, 1.00000},
 						{0.93470, 0.26240, 0.00000},
 						{0.93470, 0.26240, 1.00000},
 						{0.36840, 0.96320, 0.50000},
 						{0.53680, 0.86840, 0.00000},
 						{0.53680, 0.86840, 1.00000},
 						{0.86840, 0.53680, 0.00000},
 						{0.96320, 0.36840, 0.50000},
 						{0.86840, 0.53680, 1.00000},
 						{0.03680, 0.63160, 0.50000},
 						{0.13160, 0.46320, 1.00000},
 						{0.13160, 0.46320, 0.00000},
 						{0.46320, 0.13160, 0.00000},
 						{0.63160, 0.03680, 0.50000},
 						{0.46320, 0.13160, 1.00000},
 						{0.31770, 0.68230, 0.75240},
 						{0.31770, 0.68230, 0.24760},
 						{0.81770, 0.81770, 0.25240},
 						{0.81770, 0.81770, 0.74760},
 						{0.68230, 0.31770, 0.75240},
 						{0.68230, 0.31770, 0.24760},
 						{0.18230, 0.18230, 0.74760},
 						{0.18230, 0.18230, 0.25240},
 						{0.10190, 0.89810, 0.50000},
 						{0.60190, 0.60190, 1.00000},
 						{0.60190, 0.60190, 0.00000},
 						{0.39810, 0.39810, 1.00000},
 						{0.39810, 0.39810, 0.00000},
 						{0.89810, 0.10190, 0.50000}};
 	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
 
	r0sqd=pow(0.03*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2];  

		counter=0;
		for(m=0;m<47;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

      	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);      
    } 
    fclose(fp);  

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}    
}


void init_Z_primitive(double *phi_min, double *phi_maj)
{
	int i,j,k,m,counter;
	long ijk;
	double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd;
	double temp;
	double Coor[20][3]={{0.00000, 1.00000, 0.75000},
					    {0.00000, 1.00000, 0.25000},
					    {0.00000, 0.00000, 0.75000},
					    {0.00000, 0.00000, 0.25000},
					    {1.00000, 1.00000, 0.75000},
					    {1.00000, 1.00000, 0.25000},
					    {1.00000, 0.00000, 0.75000},
					    {1.00000, 0.00000, 0.25000},
					    {0.50000, 1.00000, 0.00000},
					    {0.00000, 0.50000, 0.00000},
					    {0.50000, 1.00000, 1.00000},
					    {0.00000, 0.50000, 1.00000},
					    {0.50000, 0.50000, 1.00000},
					    {0.50000, 0.50000, 0.00000},
					    {1.00000, 0.50000, 1.00000},
					    {0.50000, 0.00000, 1.00000},
					    {1.00000, 0.50000, 0.00000},
					    {0.50000, 0.00000, 0.00000},
					    {0.33333, 0.66667, 0.50000},
					    {0.66667, 0.33333, 0.50000}};
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
  
	r0sqd=pow(0.02*A_matrix[0]*A_matrix[4]*A_matrix[8]*0.25/Pi,2.0/3); 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

		counter=0;
		for(m=0;m<20;m++)
		{
			rx0=Coor[m][0]*A_matrix[0]+Coor[m][1]*A_matrix[3]+Coor[m][2]*A_matrix[6];
	    	ry0=Coor[m][1]*A_matrix[4]+Coor[m][2]*A_matrix[7];
		  	rz0=Coor[m][2]*A_matrix[8];
	    	rsqd=(rx-rx0)*(rx-rx0)+(ry-ry0)*(ry-ry0)+(rz-rz0)*(rz-rz0);
	    	if(rsqd<=r0sqd)    
		  	{
			  	phi_min[ijk]=1.0;
		      	phi_maj[ijk]=0.0;

	      		counter=1;
	      		break;
		  	}
		}

	  	if(counter==0)    
	  	{
	  		phi_min[ijk]=0.0;
      		phi_maj[ijk]=1.0;
	  	}

  		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
    } 
    fclose(fp);

    for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	} 
}


void init_Gyroid_primitive(double *phi_min, double *phi_maj)
{
  	int count_A;
  	double target_ratio,ratio_A,epsl;
  	int i,j,k;
	long ijk;
	double rx, ry, rz, rsqd, r0sqd;
	double dva[3],dvb[3],dvc[3];
	double da1_original[3], da2_original[3], da3_original[3];
	double krx, kry, krz;
	double pv_original[3][3]={{-0.5,0.5,0.5}, {0.5,-0.5,0.5}, {0.5,0.5,-0.5}};
	double cp_ratio[3]={2/sqrt(3),2/sqrt(3),2/sqrt(3)};

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(i=0;i<3;i++)
	{
		da1_original[i]=pv_original[0][i]*cp_ratio[0]*A_matrix[0]/(double)(Na);
		da2_original[i]=pv_original[1][i]*cp_ratio[1]*A_matrix[0]/(double)(Nb);
		da3_original[i]=pv_original[2][i]*cp_ratio[2]*A_matrix[0]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");

	r0sqd=-0.5;  
  	count_A=0;
  
  	epsl=1.0;

  	target_ratio=0.4;
  
	while(epsl>0.0005)
	{
		count_A=0;

		for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);
			krx=((i+0.5)*da1_original[0]+(j+0.5)*da2_original[0]+(k+0.5)*da3_original[0])*2.0*Pi/(cp_ratio[0]*A_matrix[0]);
			kry=((i+0.5)*da1_original[1]+(j+0.5)*da2_original[1]+(k+0.5)*da3_original[1])*2.0*Pi/(cp_ratio[1]*A_matrix[0]);
			krz=((i+0.5)*da1_original[2]+(j+0.5)*da2_original[2]+(k+0.5)*da3_original[2])*2.0*Pi/(cp_ratio[2]*A_matrix[0]);
			rsqd=sin(krx)*cos(kry)+sin(kry)*cos(krz)+sin(krz)*cos(krx);
			if(rsqd<=r0sqd)
			{
				phi_min[ijk]=1.0;
				count_A+=1;		
			}
			else
			{
				phi_min[ijk]=0.0;
			}
		}

		ratio_A=1.0*count_A/NaNbNc;
		if(ratio_A>=target_ratio)
		{
			epsl=ratio_A-target_ratio;
			r0sqd-=0.001;
		}
		else
		{
			epsl=target_ratio-ratio_A;
			r0sqd+=0.001;
		}
	} 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
  	{
	  	ijk=(long)((k*Nb+j)*(Na+2)+i);  
      
    	phi_maj[ijk]=1.0-phi_min[ijk];

    	rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

    	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);   
  	}
  	fclose(fp);

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_DoubleGyroid_primitive(double *phi_min, double *phi_maj)
{
  	int count_A;
  	double target_ratio,ratio_A,epsl;
  	int i,j,k;
	long ijk;
	double rx, ry, rz, rsqd, r0sqd;
	double dva[3],dvb[3],dvc[3];
	double da1_original[3], da2_original[3], da3_original[3];
	double krx, kry, krz;
	double pv_original[3][3]={{-0.5,0.5,0.5}, {0.5,-0.5,0.5}, {0.5,0.5,-0.5}};
	double cp_ratio[3]={2/sqrt(3),2/sqrt(3),2/sqrt(3)};

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(i=0;i<3;i++)
	{
		da1_original[i]=pv_original[0][i]*cp_ratio[0]*A_matrix[0]/(double)(Na);
		da2_original[i]=pv_original[1][i]*cp_ratio[1]*A_matrix[0]/(double)(Nb);
		da3_original[i]=pv_original[2][i]*cp_ratio[2]*A_matrix[0]/(double)(Nc);
	}	

  	FILE *fp=fopen("phi_min.dat","w");
 
	r0sqd=-0.5;  
  	count_A=0;
  
  	epsl=1.0;

  	target_ratio=0.4;
  
	while(epsl>0.0005)
	{
		count_A=0;

		for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);
			krx=((i+0.5)*da1_original[0]+(j+0.5)*da2_original[0]+(k+0.5)*da3_original[0])*2.0*Pi/(cp_ratio[0]*A_matrix[0]);
			kry=((i+0.5)*da1_original[1]+(j+0.5)*da2_original[1]+(k+0.5)*da3_original[1])*2.0*Pi/(cp_ratio[1]*A_matrix[0]);
			krz=((i+0.5)*da1_original[2]+(j+0.5)*da2_original[2]+(k+0.5)*da3_original[2])*2.0*Pi/(cp_ratio[2]*A_matrix[0]);
			rsqd=sin(krx)*cos(kry)+sin(kry)*cos(krz)+sin(krz)*cos(krx);
			if(rsqd<=r0sqd||rsqd>=-r0sqd)
			{
				phi_min[ijk]=1.0;
				count_A+=1;		
			}
			else
			{
				phi_min[ijk]=0.0;
			}
		}

		ratio_A=1.0*count_A/NaNbNc;
		if(ratio_A>=target_ratio)
		{
			epsl=ratio_A-target_ratio;
			r0sqd-=0.001;
		}
		else
		{
			epsl=target_ratio-ratio_A;
			r0sqd+=0.001;
		}
	} 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
  	{
	  	ijk=(long)((k*Nb+j)*(Na+2)+i);  
      
    	phi_maj[ijk]=1.0-phi_min[ijk];

    	rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

    	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);  
  	}
  	fclose(fp);

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_Diamond_primitive(double *phi_min, double *phi_maj)
{
  	int count_A;
  	double target_ratio,ratio_A,epsl;
  	int i,j,k;
	long ijk;
	double rx, ry, rz, rsqd, r0sqd;
	double dva[3],dvb[3],dvc[3];
	double da1_original[3], da2_original[3], da3_original[3];
	double krx, kry, krz;
	double pv_original[3][3]={{0.0,0.5,0.5}, {0.5,0.0,0.5}, {0.5,0.5,0.0}};
	double cp_ratio[3]={sqrt(2),sqrt(2),sqrt(2)};

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(i=0;i<3;i++)
	{
		da1_original[i]=pv_original[0][i]*cp_ratio[0]*A_matrix[0]/(double)(Na);
		da2_original[i]=pv_original[1][i]*cp_ratio[1]*A_matrix[0]/(double)(Nb);
		da3_original[i]=pv_original[2][i]*cp_ratio[2]*A_matrix[0]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");

	r0sqd=-0.6;  
  	count_A=0;
  
  	epsl=1.0;

  	target_ratio=0.4;
  
	while(epsl>0.0005)
	{
		count_A=0;

		for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);
			krx=((i+0.5)*da1_original[0]+(j+0.5)*da2_original[0]+(k+0.5)*da3_original[0])*2.0*Pi/(cp_ratio[0]*A_matrix[0]);
			kry=((i+0.5)*da1_original[1]+(j+0.5)*da2_original[1]+(k+0.5)*da3_original[1])*2.0*Pi/(cp_ratio[1]*A_matrix[0]);
			krz=((i+0.5)*da1_original[2]+(j+0.5)*da2_original[2]+(k+0.5)*da3_original[2])*2.0*Pi/(cp_ratio[2]*A_matrix[0]);
			rsqd=cos(krx)*cos(kry)*cos(krz)+sin(krx)*sin(kry)*sin(krz);
			if(rsqd<=r0sqd)
			{
				phi_min[ijk]=1.0;
				count_A+=1;		
			}
			else
			{
				phi_min[ijk]=0.0;
			}
		}

		ratio_A=1.0*count_A/NaNbNc;
		if(ratio_A>=target_ratio)
		{
			epsl=ratio_A-target_ratio;
			r0sqd-=0.001;
		}
		else
		{
			epsl=target_ratio-ratio_A;
			r0sqd+=0.001;
		}
	} 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
  	{
	  	ijk=(long)((k*Nb+j)*(Na+2)+i);  
      
    	phi_maj[ijk]=1.0-phi_min[ijk];

    	rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

    	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);  
  	}
  	fclose(fp);	

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_DoubleDiamond_primitive(double *phi_min, double *phi_maj)
{
  	int count_A;
  	double target_ratio,ratio_A,epsl;
  	int i,j,k;
	long ijk;
	double rx, ry, rz, rsqd, r0sqd;
	double dva[3],dvb[3],dvc[3];
	double da1_original[3], da2_original[3], da3_original[3];
	double krx, kry, krz;
	double pv_original[3][3]={{0.0,0.5,0.5}, {0.5,0.0,0.5}, {0.5,0.5,0.0}};
	double cp_ratio[3]={sqrt(2),sqrt(2),sqrt(2)};

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(i=0;i<3;i++)
	{
		da1_original[i]=pv_original[0][i]*cp_ratio[0]*A_matrix[0]/(double)(Na);
		da2_original[i]=pv_original[1][i]*cp_ratio[1]*A_matrix[0]/(double)(Nb);
		da3_original[i]=pv_original[2][i]*cp_ratio[2]*A_matrix[0]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");
 
	r0sqd=-0.5;  
  	count_A=0;
  
  	epsl=1.0;

  	target_ratio=0.6;
  
	while(epsl>0.0005)
	{
		count_A=0;

		for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);
			krx=((i+0.5)*da1_original[0]+(j+0.5)*da2_original[0]+(k+0.5)*da3_original[0])*2.0*Pi/(cp_ratio[0]*A_matrix[0]);
			kry=((i+0.5)*da1_original[1]+(j+0.5)*da2_original[1]+(k+0.5)*da3_original[1])*2.0*Pi/(cp_ratio[1]*A_matrix[0]);
			krz=((i+0.5)*da1_original[2]+(j+0.5)*da2_original[2]+(k+0.5)*da3_original[2])*2.0*Pi/(cp_ratio[2]*A_matrix[0]);
			rsqd=cos(krx)*cos(kry)*cos(krz)+sin(krx)*sin(kry)*sin(krz);
			if(rsqd<=r0sqd||rsqd>=-r0sqd)
			{
				phi_min[ijk]=1.0;
				count_A+=1;		
			}
			else
			{
				phi_min[ijk]=0.0;
			}
		}

		ratio_A=1.0*count_A/NaNbNc;
		if(ratio_A>=target_ratio)
		{
			epsl=ratio_A-target_ratio;
			r0sqd-=0.001;
		}
		else
		{
			epsl=target_ratio-ratio_A;
			r0sqd+=0.001;
		}
	} 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
  	{
	  	ijk=(long)((k*Nb+j)*(Na+2)+i);  
      
    	phi_maj[ijk]=1.0-phi_min[ijk];

    	rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

    	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]);     
  	}
  	fclose(fp);	

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_O70_primitive(double *phi_min, double *phi_maj)
{
  	int count_A;
  	double target_ratio,ratio_A,epsl;
  	int i,j,k;
	long ijk;
	double rx, ry, rz, rsqd, r0sqd;
	double dva[3],dvb[3],dvc[3];
	double da1_original[3], da2_original[3], da3_original[3];
	double krx, kry, krz;
	double pv_original[3][3]={{1.0,0.0,0.0}, {0.0,1.0,0.0}, {0.0,0.0,1.0}};
	double cp_ratio[3]={1.0,1.0,1.0};

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(i=0;i<3;i++)
	{
		da1_original[i]=pv_original[0][i]*cp_ratio[0]*A_matrix[0]/(double)(Na);
		da2_original[i]=pv_original[1][i]*cp_ratio[1]*A_matrix[0]/(double)(Nb);
		da3_original[i]=pv_original[2][i]*cp_ratio[2]*A_matrix[0]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");

	r0sqd=-0.5;  
  	count_A=0;
  
  	epsl=1.0;

  	target_ratio=0.4;
  
	while(epsl>0.0005)
	{
		count_A=0;

		for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);
			krx=((i+0.5)*da1_original[0]+(j+0.5)*da2_original[0]+(k+0.5)*da3_original[0])*2.0*Pi/(cp_ratio[0]*A_matrix[0]);
			kry=((i+0.5)*da1_original[1]+(j+0.5)*da2_original[1]+(k+0.5)*da3_original[1])*2.0*Pi/(cp_ratio[1]*A_matrix[0]);
			krz=((i+0.5)*da1_original[2]+(j+0.5)*da2_original[2]+(k+0.5)*da3_original[2])*2.0*Pi/(cp_ratio[2]*A_matrix[0]);
			rsqd=cos(krx)*cos(kry)*cos(krz)+sin(krx)*sin(kry)*cos(krz)+sin(krx)*cos(kry)*sin(krz)+cos(krx)*sin(kry)*sin(krz);
			if(rsqd<=r0sqd)
			{
				phi_min[ijk]=1.0;
				count_A+=1;		
			}
			else
			{
				phi_min[ijk]=0.0;
			}
		}

		ratio_A=1.0*count_A/NaNbNc;
		if(ratio_A<=target_ratio)
		{
			epsl=ratio_A-target_ratio;
			r0sqd-=0.001;
		}
		else
		{
			epsl=target_ratio-ratio_A;
			r0sqd+=0.001;
		}
	} 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
  	{
	  	ijk=(long)((k*Nb+j)*(Na+2)+i);  
      
    	phi_maj[ijk]=1.0-phi_min[ijk];

    	rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

    	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]); 
  	}
  	fclose(fp);

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void init_DoublePlumberNightmare_primitive(double *phi_min, double *phi_maj)
{
	int count_A;
  	double target_ratio,ratio_A,epsl;
  	int i,j,k;
	long ijk;
	double rx, ry, rz, rsqd, r0sqd;
	double dva[3],dvb[3],dvc[3];
	double da1_original[3], da2_original[3], da3_original[3];
	double krx, kry, krz;
	double pv_original[3][3]={{1.0,0.0,0.0}, {0.0,1.0,0.0}, {0.0,0.0,1.0}};
	double cp_ratio[3]={1.0,1.0,1.0};

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(i=0;i<3;i++)
	{
		da1_original[i]=pv_original[0][i]*cp_ratio[0]*A_matrix[0]/(double)(Na);
		da2_original[i]=pv_original[1][i]*cp_ratio[1]*A_matrix[0]/(double)(Nb);
		da3_original[i]=pv_original[2][i]*cp_ratio[2]*A_matrix[0]/(double)(Nc);
	}

  	FILE *fp=fopen("phi_min.dat","w");

	r0sqd=-0.8;  
  	count_A=0;
  
  	epsl=1.0;

  	target_ratio=0.4;
  
	while(epsl>0.0005&&r0sqd<-0.5&&r0sqd>-0.9)
	{
		count_A=0;

		for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);
			krx=((i+0.5)*da1_original[0]+(j+0.5)*da2_original[0]+(k+0.5)*da3_original[0])*2.0*Pi/(cp_ratio[0]*A_matrix[0]);
			kry=((i+0.5)*da1_original[1]+(j+0.5)*da2_original[1]+(k+0.5)*da3_original[1])*2.0*Pi/(cp_ratio[1]*A_matrix[0]);
			krz=((i+0.5)*da1_original[2]+(j+0.5)*da2_original[2]+(k+0.5)*da3_original[2])*2.0*Pi/(cp_ratio[2]*A_matrix[0]);
			rsqd=cos(krx)+cos(kry)+cos(krz);
			if(rsqd<=r0sqd||rsqd>=-r0sqd)
			{
				phi_min[ijk]=1.0;
				count_A+=1;		
			}
			else
			{
				phi_min[ijk]=0.0;
			}
		}

		ratio_A=1.0*count_A/NaNbNc;
		if(ratio_A<=target_ratio)
		{
			epsl=ratio_A-target_ratio;
			r0sqd-=0.001;
		}
		else
		{
			epsl=target_ratio-ratio_A;
			r0sqd+=0.001;
		}
	} 

  	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
  	{
	  	ijk=(long)((k*Nb+j)*(Na+2)+i);  
      
    	phi_maj[ijk]=1.0-phi_min[ijk];

    	rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2]; 

    	fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phi_min[ijk]); 
  	}
  	fclose(fp);

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);	

			  	phi_min[ijk]=0.0;
			  	phi_maj[ijk]=0.0;
			}
		}
	}
}


void finalize_initialization(double *phA,double *phB,double *phC,int inv)
{
	int i,j,k;
	long ijk;
	double temp;

	if(inv==0)
	{

	}
	else
	{
		for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
		{
			ijk=(long)((k*Nb+j)*(Na+2)+i);

			temp=phA[ijk];
			phA[ijk]=phB[ijk];
			phB[ijk]=temp;
		}
	}

	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);
		
		phB[ijk]/=2.0;
		phC[ijk]=phB[ijk];
	}	
}


void init_w_from_phi(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC)
{
	int i,j,k;
	long ijk;

	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);

		wA[ijk]=chiAB*N*phB[ijk]+chiAC*N*phC[ijk];
		wB[ijk]=chiAB*N*phA[ijk]+chiBC*N*phC[ijk]; 
		wC[ijk]=chiAC*N*phA[ijk]+chiBC*N*phB[ijk];
	}

	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	
			  	wA[ijk]=0.0;
			  	wB[ijk]=0.0;
			  	wC[ijk]=0.0;
			}
		}
	}
}


void init_lattice_parameters()
{
	if(in_Phase==2)		// for HEX //
	{
		A_matrix[4]=(sqrt(3.0)/2.0)*A_matrix[0];
		A_matrix[8]=1.0*A_matrix[0];
		A_matrix[3]=-0.5*A_matrix[0];     
		A_matrix[7]=0.0*A_matrix[0];
		A_matrix[6]=0.0*A_matrix[0];
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==3||in_Phase==6||in_Phase==7)		// for BCC, G and DG//
	{
		A_matrix[4]=0.9428090415820635*A_matrix[0];
		A_matrix[8]=0.8164965809277264*A_matrix[0];
		A_matrix[3]=-0.3333333333333333*A_matrix[0];     
		A_matrix[7]=-0.4714045207910315*A_matrix[0];
		A_matrix[6]=-0.3333333333333333*A_matrix[0];
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==1003)		// for BCC conventional cell //
	{
		A_matrix[4]=A_matrix[0];
		A_matrix[8]=A_matrix[0];
		A_matrix[3]=0.0*A_matrix[0];     
		A_matrix[7]=0.0*A_matrix[0];
		A_matrix[6]=0.0*A_matrix[0];
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==4||in_Phase==8||in_Phase==9)		// for FCC, D, DD //
	{
		A_matrix[4]=0.8660254037844385*A_matrix[0];
		A_matrix[8]=0.816496580927726*A_matrix[0];
		A_matrix[3]=0.5*A_matrix[0];     
		A_matrix[7]=0.28867513459481275*A_matrix[0];
		A_matrix[6]=0.5*A_matrix[0];
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==5)	// for HCP //
	{
		A_matrix[4]=(sqrt(3.0)/2.0)*A_matrix[0];
		A_matrix[8]=sqrt(8.0/3.0)*A_matrix[0];
		A_matrix[3]=-0.5*A_matrix[0];      
		A_matrix[7]=0.0*A_matrix[0];
		A_matrix[6]=0.0*A_matrix[0];
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==10)	// for C14 //
	{
		A_matrix[4]=(sqrt(3.0)/2.0)*A_matrix[0];
		A_matrix[8]=1.6408733959*A_matrix[0];
		A_matrix[3]=-0.5*A_matrix[0];     
		A_matrix[7]=0.0*A_matrix[0];
		A_matrix[6]=0.0*A_matrix[0];
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==13)	// for Sigma //
	{
		A_matrix[4]=A_matrix[0];
		A_matrix[8]=A_matrix[0]/1.89;
		A_matrix[3]=0.0;     
		A_matrix[7]=0.0;
		A_matrix[6]=0.0;
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==14)	// for Z //
	{
		A_matrix[4]=(sqrt(3.0)/2.0)*A_matrix[0];
		A_matrix[8]=0.99208540401*A_matrix[0];
		A_matrix[3]=-0.5*A_matrix[0];     
		A_matrix[7]=0.0*A_matrix[0];
		A_matrix[6]=0.0*A_matrix[0];
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else if(in_Phase==15)	// O17 //
	{
		A_matrix[4]=2.0*A_matrix[0];
		A_matrix[8]=2.0*sqrt(3.0)*A_matrix[0];
		A_matrix[3]=0.0;     
		A_matrix[7]=0.0;
		A_matrix[6]=0.0;
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}
	else	// cubic box for others //
	{
		A_matrix[4]=A_matrix[0];
		A_matrix[8]=A_matrix[0];
		A_matrix[3]=0.0;     
		A_matrix[7]=0.0;
		A_matrix[6]=0.0;
		fix_para[0]=A_matrix[3]/A_matrix[4];
		fix_para[1]=A_matrix[7]/A_matrix[8];
		fix_para[2]=A_matrix[6]/A_matrix[8];
	}

  	A_matrix[1]=0.0;							
	A_matrix[2]=0.0;
	A_matrix[5]=0.0;
}


void discretize_space()
{
	if(in_Phase==1||in_Phase==0)	//for dis and lamella//
	{
  		Na=64;
		Nb=16;
		Nc=16;            
	}
	else if(in_Phase==2)	//for Hex//
	{
  		Na=48;
		Nb=48;
		Nc=16;             
	}
	else if(in_Phase==3||in_Phase==4||in_Phase==5)	//for BCC, FCC, HCP//
	{
  		Na=48;
		Nb=48;
		Nc=48;             
	}
	else if(in_Phase==11||in_Phase==12)		//for C15 and A15//
	{
  		Na=96;
		Nb=96;
		Nc=96;            
	}
	else if(in_Phase==13)	//for Sigma// 
	{
		Na=128;
		Nb=128;
		Nc=64;            
	}
	else if(in_Phase==15)	//for O70// 
	{
		Na=32;
		Nb=48;
		Nc=64;            
	}
	else	//for other phases//
	{
		Na=64;
		Nb=64;
		Nc=64;
	}
}


void write_ph_init_point(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC)
{
	int i,j,k;
	long ijk;
	double rx,ry,rz;
	FILE *fp=fopen("phi_init_point.dat","w");

  	for(k=0;k<Nc;k++)
  	{
  		for(j=0;j<Nb;j++)
  		{
  			for(i=0;i<Na;i++)
  			{
  				ijk=(long)((k*Nb+j)*(Na+2)+i);				
  				fprintf(fp,"%lf %lf %lf %lf %lf %lf\n",phA[ijk],phB[ijk],phC[ijk],wA[ijk],wB[ijk],wC[ijk]);
  			}
  		}
  	}		
  	fclose(fp);

  	fp=fopen("phA_init_point.dat","w");
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);	

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2];

		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phA[ijk]);
	}		
	fclose(fp);
}


void reading_init_point(double *wA,double *wB,double *wC,double *phA,double *phB,double *phC)
{
  	int i,j,k;
	long ijk;
  	FILE *fp;
  	fp=fopen("phi_init_point.dat","r");
  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=0;i<Na;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	fscanf(fp,"%lf %lf %lf %lf %lf %lf\n",&phA[ijk],&phB[ijk],&phC[ijk],&wA[ijk],&wB[ijk],&wC[ijk]);
			}
		}
	}		
  	fclose(fp); 

  	for(k=0;k<Nc;k++)
	{
	  	for(j=0;j<Nb;j++)
		{
		   	for(i=Na;i<Na+2;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);			
			  	phA[ijk]=0.0;
			  	phB[ijk]=0.0;
			  	phC[ijk]=0.0;
			  	wA[ijk]=0.0;
			  	wB[ijk]=0.0;
			  	wC[ijk]=0.0;
			}
		}
	}
}


//********************Output morphologies******************************
void write_ph(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC)
{
	int i,j,k;
	long ijk;
	FILE *fp=fopen("phi.dat","w");

  	for(k=0;k<Nc;k++)
  	{
  		for(j=0;j<Nb;j++)
  		{
  			for(i=0;i<Na;i++)
  			{
  				ijk=(long)((k*Nb+j)*(Na+2)+i);				
  				fprintf(fp,"%lf %lf %lf %lf %lf %lf\n",phA[ijk],phB[ijk],phC[ijk],wA[ijk],wB[ijk],wC[ijk]);
  			}
  		}
  	}		
  	fclose(fp);
}


void write_phA(double *phA)
{
	int i,j,k;
	long ijk;
	FILE *fp=fopen("phA.dat","w");
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);	

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2];

		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phA[ijk]);
	}		
	fclose(fp);
}


void write_phB(double *phB)
{
	int i,j,k;
	long ijk;
	FILE *fp=fopen("phB.dat","w");
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);	

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2];

		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phB[ijk]);
	}		
	fclose(fp);
}


void write_phC(double *phC)
{
	int i,j,k;
	long ijk;
	FILE *fp=fopen("phC.dat","w");
	double dva[3],dvb[3],dvc[3];

	for(i=0;i<3;i++)
	{
		dva[i]=A_matrix[i]/(double)(Na);
		dvb[i]=A_matrix[i+3]/(double)(Nb);
		dvc[i]=A_matrix[i+6]/(double)(Nc);
	}

	for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
	{
		ijk=(long)((k*Nb+j)*(Na+2)+i);	

		rx=(i+0.5)*dva[0]+(j+0.5)*dvb[0]+(k+0.5)*dvc[0];
		ry=(j+0.5)*dvb[1]+(k+0.5)*dvc[1];
		rz=(k+0.5)*dvc[2];

		fprintf(fp,"%.6lf %.6lf %.6lf %.6lf\n", rx, ry, rz, phC[ijk]);
	}		
	fclose(fp);
}


int output(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC,double scan_para,struct result_struct results,char filename[])
{
	double i,j,k;
	long ijk;
	double freeEnergy, inCompMax, freeDiff, err, stress_max;
	double strict_tolerance=1e-5, warning_tolerance=5e-6;
	int diagnosis=1;
	FILE *fp;

	freeEnergy=results.freeEnergy;
	inCompMax=results.inCompMax;
	freeDiff=results.freeDiff;
	err=results.err;
	stress_max=results.stress_max;

	if(world_rank==0)
  	{
  		// Disorder check:
  		if(in_Phase!=0) 
  		{
  			double phA_ave=0.0, phB_ave=0.0, phC_ave=0.0, phA_std=0.0, phB_std=0.0, phC_std=0.0;
  			for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);
				phA_ave+=phA[ijk];
				phB_ave+=phB[ijk];
				phC_ave+=phC[ijk];
			}
			phA_ave/=NaNbNc;
			phB_ave/=NaNbNc;
			phC_ave/=NaNbNc;

			for(k=0;k<Nc;k++)for(j=0;j<Nb;j++)for(i=0;i<Na;i++)
			{
				ijk=(long)((k*Nb+j)*(Na+2)+i);
				phA_std+=(phA[ijk]-phA_ave)*(phA[ijk]-phA_ave);
				phB_std+=(phB[ijk]-phB_ave)*(phB[ijk]-phB_ave);
				phC_std+=(phC[ijk]-phC_ave)*(phC[ijk]-phC_ave);
			}
			phA_std/=NaNbNc;
			phB_std/=NaNbNc;
			phC_std/=NaNbNc;

			if(phA_std<1e-5&&phB_std<1e-5&&phC_std<1e-5)
			{
				fp=fopen(filename,"a");
				fprintf(fp, "Suspect converging to Disordered!\n");
				fclose(fp);

				diagnosis=0;
			}
  		}

  		// Convergence check:
  		if(diagnosis!=0)
  		{
  			if(inCompMax>strict_tolerance||freeDiff>strict_tolerance||err>strict_tolerance||stress_max>strict_tolerance)
	  		{
	  			fp=fopen(filename,"a");
				fprintf(fp, "Failed to converge!\n");
				fclose(fp);

				diagnosis=0;
	  		}
  		}
  		
  		if(diagnosis!=0)
  		{
	  		write_ph(phA,phB,phC,wA,wB,wC);
			write_phA(phA);
			write_phB(phB);
			write_phC(phC);

			fp=fopen(filename,"a");
			fprintf(fp, "%lf\t%.7e\t", scan_para, freeEnergy);

	  		if((inCompMax>warning_tolerance||freeDiff>warning_tolerance||err>warning_tolerance||stress_max>warning_tolerance)&&(inCompMax<strict_tolerance&&freeDiff<strict_tolerance&&err<strict_tolerance&&stress_max<strict_tolerance))
	  		{
				fprintf(fp, "Warning: ");
				if(inCompMax>warning_tolerance)
				{
					fprintf(fp, "inCompMax=%.4e ", inCompMax);
				}
				if(freeDiff>warning_tolerance)
				{
					fprintf(fp, "freeDiff=%.4e ", freeDiff);
				}
				if(err>warning_tolerance)
				{
					fprintf(fp, "err=%.4e ", err);
				}

				if(stress_max>warning_tolerance)
				{
					fprintf(fp, "stress -> ");
					if(fabs(stress[0])>warning_tolerance)
					{
						fprintf(fp, "a1x=%.4e ", stress[0]);
					}
					if(fabs(stress[3])>warning_tolerance)
					{
						fprintf(fp, "a2x=%.4e ", stress[3]);
					}
					if(fabs(stress[1])>warning_tolerance)
					{
						fprintf(fp, "a2y=%.4e ", stress[1]);
					}
					if(fabs(stress[5])>warning_tolerance)
					{
						fprintf(fp, "a3x=%.4e ", stress[5]);
					}
					if(fabs(stress[4])>warning_tolerance)
					{
						fprintf(fp, "a3y=%.4e ", stress[4]);
					}
					if(fabs(stress[2])>warning_tolerance)
					{
						fprintf(fp, "a3z=%.4e ", stress[2]);
					}
				}
	  		}

			fprintf(fp, "\n");
			fclose(fp);	
	  	}
	}

	MPI_Bcast(&diagnosis, 1, MPI_INT, 0, MPI_COMM_WORLD);

  	return diagnosis;
}


void Distribute(double *wA,double *wB,double *wC)
{
	int i,j,k;

	if(world_rank == 0) 
  	{
  		for(i=1;i<world_size;i++)
  		{
  			MPI_Send(&wA[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
  			MPI_Send(&wB[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
  			MPI_Send(&wC[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
  		}
  	}
  	else
  	{
  		MPI_Recv(wA, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  		MPI_Recv(wB, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  		MPI_Recv(wC, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  	}
  	MPI_Barrier(MPI_COMM_WORLD);
}


void Gather(double *phA,double *phB,double *phC,double *wA,double *wB,double *wC)
{
	int i,j,k;

	if(world_rank == 0) 
  	{	
  		for(i=1;i<world_size;i++)
  		{
	  		MPI_Recv(&phA[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  		MPI_Recv(&phB[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  		MPI_Recv(&phC[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  		MPI_Recv(&wA[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  		MPI_Recv(&wB[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  		MPI_Recv(&wC[i*local_Na_2_NbNc], local_Na_2_NbNc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  		}		
  	}
  	else 
  	{
  		MPI_Send(phA, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(phB, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(phC, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(wA, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(wB, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(wC, local_Na_2_NbNc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  	}
  	MPI_Barrier(MPI_COMM_WORLD);
}







