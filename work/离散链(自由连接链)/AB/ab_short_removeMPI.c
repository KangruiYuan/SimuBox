/************************************************************************/
/*
AB diblock system modeled by freely-jointed chain with finite
range interaction

3D lattice parameter optimization

remove MPI
*/
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include<mpi.h>
// #include<fftw3-mpi.h>
#include <fftw3.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#define MaxIT 2000 // Maximum iteration steps

#define sigma (sqrt(3.0) * blA)
#define blA 1.0
#define blB (1.0 / 1.32)

#define nd 3

#define Pi 3.141592653589

// Parameters used in Anderson convergence //
#define N_hist 30
//

void init_random_W(double *wA, double *wB);
void init_reading(double *wA, double *wB, double *phA, double *phB);
void init_Disordered(double *wA, double *wB, double *phA, double *phB);
void init_Lamella(double *wA, double *wB, double *phA, double *phB);
void init_HEX(double *wA, double *wB, double *phA, double *phB, int inv);
void init_BCC(double *wA, double *wB, double *phA, double *phB, int inv);
void init_FCC(double *wA, double *wB, double *phA, double *phB, int inv);
void init_HCP(double *wA, double *wB, double *phA, double *phB, int inv);
void init_Gyroid(double *wA, double *wB, double *phA, double *phB, int inv);
void init_DoubleGyroid(double *wA, double *wB, double *phA, double *phB, int inv);
void init_Diamond(double *wA, double *wB, double *phA, double *phB, int inv);
void init_DoubleDiamond(double *wA, double *wB, double *phA, double *phB, int inv);
void init_C14(double *wA, double *wB, double *phA, double *phB, int inv);
void init_C15(double *wA, double *wB, double *phA, double *phB, int inv);
void init_A15(double *wA, double *wB, double *phA, double *phB, int inv);
void init_Sigma(double *wA, double *wB, double *phA, double *phB, int inv);
void init_Z(double *wA, double *wB, double *phA, double *phB, int inv);
void init_O70(double *wA, double *wB, double *phA, double *phB, int inv);
void init_DoublePlumberNightmare(double *wA, double *wB, double *phA, double *phB, int inv);

double freeE(double *wA, double *wB, double *phA, double *phB);
double getConc(double *phA, double *phB, double *wA, double *wB);
void get_q(double *prop, fftw_complex *props, double *exp_w, double bl, double *qInt, int ns, int sign);
void conjunction_propagation(double *qInt, fftw_complex *qInts, double *exp_w, double bl);
void write_ph(double *phA, double *phB, double *wA, double *wB);
void write_phA(double *phA);
void write_phB(double *phB);

double error_cal(double *waDiffs, double *wbDiffs, double *wAs, double *wBs);
void update_flds_hist(double *waDiff, double *wbDiff, double *wAnew, double *wBnew, double *del, double *outs);
void Anderson_mixing(double *del, double *outs, int N_rec, double *wA, double *wB);

// void Distribute(double *wA, double *wB);
// void Gather(double *phA, double *phB, double *wA, double *wB);

//*****Global variables*****//
// Inverse phase?
int inverse;
// Scan parameters:
double d_hAB;
int hAB_N;
// Parameters of the polymer chain:
int NsA, NsB, N;
double hAB, f, fB;
// Target Phase:
int in_Phase;
// Period and box size:
double lx, ly, lz, *kxyz, *u_k, *sum_factor;
int Nx, Ny, Nz;
// int local_Nx_2_NyNz, local_NxNyNz1, Nx_2_NyNz;
int NxNyNz, Nxh1, NxNyNz1;
int hist_Nx;
double lattice_para[3], lattice_para_new[3];
// Initialization parameters:
double rx, ry, rz, rx0, ry0, rz0, rsqd, r0sqd, dx, dy, dz;
// FFTW:
double *in;
fftw_complex *out;
fftw_plan p_forward, p_backward;
// Stress:
double *stress, *stress_bond, *stress_FR;
double *kx_3d, *ky_3d, *kz_3d;
// Time test:
double start, end;
// MPI:
// int world_rank, world_size;
// ptrdiff_t alloc_local, local_Nz, local_Nz_start;

int main(int argc, char **argv)
{
    double *wA, *wB, *phA, *phB;
    double e1;
    int i, j, k, iseed = -3;
    int in_method;
    long ijk;
    char filename[100], phasename[100], category[100];

    FILE *fp;
    time_t ts;
    iseed = time(&ts);
    srand48(iseed);

    // Initialize the MPI environment
    // MPI_Init(&argc, &argv);
    // Find out rank, size
    // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //*****Read in parameters from a file named para*****//
    fp = fopen("para_common", "r");
    fscanf(fp, "%d", &inverse);
    fscanf(fp, "%d,%d", &N, &NsA);
    fclose(fp);

    fp = fopen("para", "r");
    fscanf(fp, "%d,%d", &in_method, &in_Phase);
    fscanf(fp, "%lf,%lf,%d", &hAB, &d_hAB, &hAB_N);
    fscanf(fp, "%lf", &lx);
    fclose(fp);

    //*****Set the name of the output file*****//
    //*****Determine the category of target phase*****//
    if (inverse == 0)
    {
        sprintf(category, "");
    }
    else
    {
        sprintf(category, "inv_");
    }

    //*****Determine the name of the target phase*****//
    if (in_Phase == 0)
    {
        sprintf(phasename, "Dis");
        sprintf(category, "");
    }
    else if (in_Phase == 1)
    {
        sprintf(phasename, "Lam");
    }
    else if (in_Phase == 2)
    {
        sprintf(phasename, "HEX");
    }
    else if (in_Phase == 3)
    {
        sprintf(phasename, "BCC");
    }
    else if (in_Phase == 4)
    {
        sprintf(phasename, "FCC");
    }
    else if (in_Phase == 5)
    {
        sprintf(phasename, "HCP");
    }
    else if (in_Phase == 6)
    {
        sprintf(phasename, "G");
    }
    else if (in_Phase == 7)
    {
        sprintf(phasename, "DG");
    }
    else if (in_Phase == 8)
    {
        sprintf(phasename, "D");
    }
    else if (in_Phase == 9)
    {
        sprintf(phasename, "DD");
    }
    else if (in_Phase == 10)
    {
        sprintf(phasename, "C14");
    }
    else if (in_Phase == 11)
    {
        sprintf(phasename, "C15");
    }
    else if (in_Phase == 12)
    {
        sprintf(phasename, "A15");
    }
    else if (in_Phase == 13)
    {
        sprintf(phasename, "Sigma");
    }
    else if (in_Phase == 14)
    {
        sprintf(phasename, "Z");
    }
    else if (in_Phase == 16)
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

    //*****Set box size according to phase*****//
    if (in_Phase == 1 || in_Phase == 0)
    {
        Nx = 64;
        Ny = 64; //64*16*16
        Nz = 4; // for lamella//
    }
    else if (in_Phase == 2)
    {
        Nx = 64;
        Ny = 64;
        Nz = 16; // for Hex//
    }
    else if (in_Phase == 5 || in_Phase == 7 || in_Phase == 9 || in_Phase == 11 || in_Phase == 12 || in_Phase == 14)
    {
        Nx = 96;
        Ny = 96;
        Nz = 96; // for HCP, C15, A15, Z//
    }
    else if (in_Phase == 10)
    {
        Nx = 64;
        Ny = 128;
        Nz = 128; // for C14//
    }
    else if (in_Phase == 13)
    {
        Nx = 128;
        Ny = 128;
        Nz = 64; // for Sigma//
    }
    else
    {
        Nx = 64;
        Ny = 64;
        Nz = 64;
    }

    /* Assign values for looping constants */
    NxNyNz = Nx * Ny * Nz;
    Nxh1 = Nx / 2 + 1;
    NxNyNz1 = Nz * Ny * Nxh1;

    /* Initialize MPI FFTW */
    // fftw_mpi_init();

    /* get local data size and allocate */
    // alloc_local = fftw_mpi_local_size_3d(Nz, Ny, Nxh1, MPI_COMM_WORLD, &local_Nz, &local_Nz_start);

    /* Assign values for looping constants needed by MPI */
    // Nx_2_NyNz = (Nx) * Ny * Nz; // Add padding space in real space used by MPI FFTW
    // local_Nx_2_NyNz = (Nx) * Ny * local_Nz;
    // local_NxNyNz1 = Nxh1 * Ny * local_Nz;
    hist_Nx = NxNyNz * 2 + nd;

    // in = fftw_alloc_real(2 * alloc_local);
    // out = fftw_alloc_complex(alloc_local);

    in = (double *) fftw_malloc(sizeof(double) * NxNyNz);
    out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * NxNyNz1);

    /* create plan for out-of-place r2c DFT */
    // p_forward = fftw_mpi_plan_dft_r2c_3d(Nz, Ny, Nx, in, out, MPI_COMM_WORLD, FFTW_MEASURE);
    // p_backward = fftw_mpi_plan_dft_c2r_3d(Nz, Ny, Nx, out, in, MPI_COMM_WORLD, FFTW_MEASURE);

    p_forward = fftw_plan_dft_r2c_3d(Nz, Ny, Nx, in, out, FFTW_MEASURE);
    p_backward = fftw_plan_dft_c2r_3d(Nz, Ny, Nx, out, in, FFTW_MEASURE);

    stress = (double *)malloc(sizeof(double) * nd);
    stress_bond = (double *)malloc(sizeof(double) * nd);
    stress_FR = (double *)malloc(sizeof(double) * nd);
    kxyz = (double *)malloc(sizeof(double) * NxNyNz1);
    u_k = (double *)malloc(sizeof(double) * NxNyNz1);
    sum_factor = (double *)malloc(sizeof(double) * NxNyNz1);
    kx_3d = (double *)malloc(sizeof(double) * NxNyNz1);
    ky_3d = (double *)malloc(sizeof(double) * NxNyNz1);
    kz_3d = (double *)malloc(sizeof(double) * NxNyNz1);

    wA =  (double *)malloc(sizeof(double) * NxNyNz);
    wB =  (double *)malloc(sizeof(double) * NxNyNz);
    phA = (double *)malloc(sizeof(double) * NxNyNz);
    phB = (double *)malloc(sizeof(double) * NxNyNz);



    // Used to do the summation over all components in Fourier space//
    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = 0; i < Nxh1; i++)
            {
                ijk = (k * Ny + j) * Nxh1 + i;
                if (i == 0 || i == Nxh1 - 1)
                {
                    sum_factor[ijk] = 1.0;
                }
                else
                {
                    sum_factor[ijk] = 2.0;
                }
            }
        }
    }


    if (in_Phase == 2)
    {
        ly = sqrt(3.0) * lx; // for HEX //
        lz = lx;
    }
    else if (in_Phase == 5)
    {
        ly = sqrt(3.0) * lx;
        lz = sqrt(8.0 / 3.0) * lx; // for HCP
    }
    else if (in_Phase == 10)
    {
        ly = sqrt(3.0) * lx;
        lz = 1.6 * lx; // for C14
    }
    else if (in_Phase == 13)
    {
        ly = lx;
        lz = lx / 1.89; // for sigma
    }
    else if (in_Phase == 14)
    {
        ly = sqrt(3.0) * lx;
        lz = 0.99 * lx; // for Z
    }
    else
    {
        ly = lx; // cubic box for others //
        lz = lx;
    }

    dx = lx / Nx;
    dy = ly / Ny;
    dz = lz / Nz;

    NsB = N - NsA;
    f = ((double)(NsA)) / ((double)(N));
    fB = 1.0 - f;


    /***************Initialize wA, wB******************/
    if (in_method == 0)
    {
        if (in_Phase == 0)
        {
            init_Disordered(wA, wB, phA, phB);
        }
        else if (in_Phase == 1)
        {
            init_Lamella(wA, wB, phA, phB);
        }
        else if (in_Phase == 2)
        {
            init_HEX(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 3)
        {
            init_BCC(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 4)
        {
            init_FCC(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 5)
        {
            init_HCP(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 6)
        {
            init_Gyroid(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 7)
        {
            init_DoubleGyroid(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 8)
        {
            init_Diamond(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 9)
        {
            init_DoubleDiamond(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 10)
        {
            init_C14(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 11)
        {
            init_C15(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 12)
        {
            init_A15(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 13)
        {
            init_Sigma(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 14)
        {
            init_Z(wA, wB, phA, phB, inverse);
        }
        else if (in_Phase == 16)
        {
            init_DoublePlumberNightmare(wA, wB, phA, phB, inverse);
        }
        else
        {
            init_random_W(wA, wB);
            printf("Random initialization is used !");
        }
    }
    else if (in_method == 1)
    {
        init_reading(wA, wB, phA, phB);
    }


    // Distribute(wA, wB);

    // printf("----0000----\n");
    // fflush(stdout);

    for (i = 0; i < hAB_N; i++)
    {
        e1 = freeE(wA, wB, phA, phB);

        // Gather(phA, phB, wA, wB);

        write_ph(phA, phB, wA, wB);
        write_phA(phA);
        write_phB(phB);

        fp = fopen("freeEnergy.dat", "a");
        fprintf(fp, "%lf\t%+.7e\n", hAB, e1);
        fclose(fp);

        hAB += d_hAB;
    }

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);

    // MPI_Finalize();

    fftw_free(in);
    fftw_free(out);

    free(wA);
    free(wB);
    free(phA);
    free(phB);
    free(kxyz);
    free(u_k);
    free(sum_factor);
    free(kx_3d);
    free(ky_3d);
    free(kz_3d);
    free(stress);
    free(stress_bond);
    free(stress_FR);

    return 1;
}

//*************************************main loop****************************************
double freeE(double *wA, double *wB, double *phA, double *phB)
{
    int i, j, k, iter, maxIter;
    long ijk;
    double kx[Nx], ky[Ny], kz[Nz];
    double freeEnergy, freeOld, parfQ;
    double freeABW, freeABW0, freeABW1, freeS, freeDiff, freeAB0, freeAB1, freeW0, freeW1;
    double Sm1, Sm2, Sm3, Sm4, wopt, psum, fpsum, lambda, *psuC;
    double *waDiff, *wbDiff, inCompMax;
    double *del, *outs, *eta, *wAnew, *wBnew, err, *dudk;
    double *stress_FR_common;
    int N_rec;
    fftw_complex *phAs, *phBs, *wAs, *wBs, *wAnews, *wBnews, *etas;
    FILE *fp;

    psuC =   (double *)malloc(sizeof(double) * NxNyNz);
    waDiff = (double *)malloc(sizeof(double) * NxNyNz);
    wbDiff = (double *)malloc(sizeof(double) * NxNyNz);

    phAs = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);
    phBs = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);

    wAs = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);
    wBs = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);
    wAnew = (double *)malloc(sizeof(double) * NxNyNz);
    wBnew = (double *)malloc(sizeof(double) * NxNyNz);
    wAnews = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);
    wBnews = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);

    eta = (double *)malloc(sizeof(double) * NxNyNz);
    etas = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);

    del = (double *)malloc(sizeof(double) * N_hist * hist_Nx);
    outs = (double *)malloc(sizeof(double) * N_hist * hist_Nx);

    dudk = (double *)malloc(sizeof(double) * NxNyNz1);

    stress_FR_common = (double *)malloc(sizeof(double) * NxNyNz1);

    Sm1 = 1e-7;
    Sm2 = 1e-8;
    Sm3 = 1e-5;
    Sm4 = 1e-5;
    maxIter = MaxIT;
    wopt = 0.05;
    psum = 0.0;
    fpsum = 0.0;
    lambda = 0.5;

    iter = 0;

    freeEnergy = 0.0;

    lattice_para[0] = lx;
    lattice_para[1] = ly;
    lattice_para[2] = lz;

    

    do
    {

        iter = iter + 1;

        if (iter > 200 && iter <= 500)
        {
            lambda = 5.0;
        }
        // else if(iter>300&&iter<=400)
        // {
        //         lambda=10.0;
        // }
        else
        {
            lambda = 10.0;
        }

        lx = lattice_para[0];
        ly = lattice_para[1];
        lz = lattice_para[2];

        // Define Fourier components//
        for (i = 0; i <= Nx / 2; i++)
            kx[i] = 2 * Pi * i * 1.0 / lx;
        for (i = Nx / 2 + 1; i < Nx; i++)
            kx[i] = 2 * Pi * (i - Nx) * 1.0 / lx;
        for (i = 0; i < Nx; i++)
            kx[i] *= kx[i];

        for (i = 0; i <= Ny / 2; i++)
            ky[i] = 2 * Pi * i * 1.0 / ly;
        for (i = Ny / 2 + 1; i < Ny; i++)
            ky[i] = 2 * Pi * (i - Ny) * 1.0 / ly;
        for (i = 0; i < Ny; i++)
            ky[i] *= ky[i];

        for (i = 0; i <= Nz / 2; i++)
            kz[i] = 2 * Pi * i * 1.0 / lz;
        for (i = Nz / 2 + 1; i < Nz; i++)
            kz[i] = 2 * Pi * (i - Nz) * 1.0 / lz;
        for (i = 0; i < Nz; i++)
            kz[i] *= kz[i];

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nxh1; i++)
                {
                    ijk = (long)((k * Ny + j) * Nxh1 + i);
                    kx_3d[ijk] = kx[i];
                }

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nxh1; i++)
                {
                    ijk = (long)((k * Ny + j) * Nxh1 + i);
                    ky_3d[ijk] = ky[j];
                }

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nxh1; i++)
                {
                    ijk = (long)((k * Ny + j) * Nxh1 + i);
                    kz_3d[ijk] = kz[k];
                }

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nxh1; i++)
                {
                    ijk = (long)((k * Ny + j) * Nxh1 + i);
                    kxyz[ijk] = sqrt(kx[i] + ky[j] + kz[k]);
                }

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            u_k[ijk] = exp(-kxyz[ijk] * kxyz[ijk] * sigma * sigma / 6.0);
        }

        // 1.Calculate propagators and then the segment concentrations//
        parfQ = getConc(phA, phB, wA, wB);

        // 2.Calculate incompresibility and inCompMax//
        inCompMax = 0.0;
        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nx; i++)
                {
                    ijk = (long)((k * Ny + j) * (Nx) + i); //?? 为什么加2
                    psum = 1.0 - phA[ijk] - phB[ijk];
                    psuC[ijk] = psum;
                    fpsum = fabs(psum);
                    if (fpsum > inCompMax)
                        inCompMax = fpsum;
                }

        // MPI_Allreduce(MPI_IN_PLACE, &inCompMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // 3.Update the presure field in real space//
        for (ijk = 0; ijk < NxNyNz; ijk++)
        {
            eta[ijk] = (wA[ijk] + wB[ijk] - hAB) / 2.0 - lambda * psuC[ijk];
        }

        // 4.Forward Fourier transferm segment concentrations and pressure field to k-space//
        // fftw_mpi_execute_dft_r2c(p_forward, eta, etas);

        // apply a given plan to a different array using the “newarray execute” functions
        fftw_execute_dft_r2c(p_forward, eta, etas);

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            etas[ijk][0] = etas[ijk][0] / NxNyNz;
            etas[ijk][1] = etas[ijk][1] / NxNyNz;
        }
        // fftw_mpi_execute_dft_r2c(p_forward, phA, phAs);
        fftw_execute_dft_r2c(p_forward, phA, phAs);

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            phAs[ijk][0] = phAs[ijk][0] / NxNyNz;
            phAs[ijk][1] = phAs[ijk][1] / NxNyNz;
        }
        // fftw_mpi_execute_dft_r2c(p_forward, phB, phBs);
        fftw_execute_dft_r2c(p_forward, phB, phBs);

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            phBs[ijk][0] = phBs[ijk][0] / NxNyNz;
            phBs[ijk][1] = phBs[ijk][1] / NxNyNz;
        }

        // 5.Obtain the output conjugate fields in k-space//
        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            wAnews[ijk][0] = hAB * u_k[ijk] * phBs[ijk][0] + etas[ijk][0];
            wAnews[ijk][1] = hAB * u_k[ijk] * phBs[ijk][1] + etas[ijk][1];
            wBnews[ijk][0] = hAB * u_k[ijk] * phAs[ijk][0] + etas[ijk][0];
            wBnews[ijk][1] = hAB * u_k[ijk] * phAs[ijk][1] + etas[ijk][1];
        }

        // 6.Backward Fourier transferm to obtain the output conjugate field in real space//
        // fftw_mpi_execute_dft_c2r(p_backward, wAnews, wAnew);
        // fftw_mpi_execute_dft_c2r(p_backward, wBnews, wBnew);

        fftw_execute_dft_c2r(p_backward, wAnews, wAnew);
        fftw_execute_dft_c2r(p_backward, wBnews, wBnew);

        // 7.Compute the deviation functions or residuals//
        for (ijk = 0; ijk < NxNyNz; ijk++)
        {
            waDiff[ijk] = wAnew[ijk] - wA[ijk];
            wbDiff[ijk] = wBnew[ijk] - wB[ijk];
        }

        // Stress//
        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            dudk[ijk] = -1 * (sigma * sigma) * u_k[ijk] / 6.0;
        }

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            stress_FR_common[ijk] = sum_factor[ijk] * dudk[ijk] * hAB * (phAs[ijk][0] * phBs[ijk][0] + phAs[ijk][1] * phBs[ijk][1]);
        }

        stress_FR[0] = 0.0;
        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            stress_FR[0] += (-2 / lx) * kx_3d[ijk] * stress_FR_common[ijk];
        }

        stress_FR[1] = 0.0;
        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            stress_FR[1] += (-2 / ly) * ky_3d[ijk] * stress_FR_common[ijk];
        }

        stress_FR[2] = 0.0;
        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            stress_FR[2] += (-2 / lz) * kz_3d[ijk] * stress_FR_common[ijk];
        }

        stress[0] = -(-stress_bond[0] + stress_FR[0]);
        stress[1] = -(-stress_bond[1] + stress_FR[1]);
        stress[2] = -(-stress_bond[2] + stress_FR[2]);
        // MPI_Allreduce(MPI_IN_PLACE, stress, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        stress[0] *= sqrt(lx * ly * lz);
        stress[1] *= sqrt(lx * ly * lz);
        stress[2] *= sqrt(lx * ly * lz);
        lattice_para_new[0] = lattice_para[0] + stress[0];
        lattice_para_new[1] = lattice_para[1] + stress[1];
        lattice_para_new[2] = lattice_para[2] + stress[2];

        // 8.Apply either simple mixing or Anderson mixing//
        // judge the error
        err = error_cal(waDiff, wbDiff, wA, wB);
        // update the history fields, and zero is new fields
        update_flds_hist(waDiff, wbDiff, wAnew, wBnew, del, outs);

        if (iter < 150)
        {
            for (ijk = 0; ijk < NxNyNz; ijk++)
            {
                wA[ijk] += wopt * waDiff[ijk];
                wB[ijk] += wopt * wbDiff[ijk];
            }

            lattice_para[0] += wopt * stress[0];
            lattice_para[1] += wopt * stress[1];
            lattice_para[2] += wopt * stress[2];
        }
        else
        {

            if (iter == 1 || iter % 5 == 0 || iter >= maxIter)
            {
                FILE *fp = fopen("ing.dat", "a");
                fprintf(fp, "/***** enter Anderson mixing *****/\n");
                fclose(fp);
            }

            N_rec = (iter - 1) < N_hist ? (iter - 1) : N_hist;
            Anderson_mixing(del, outs, N_rec, wA, wB);
        }

        // 9.Calculate the free energy density in k space//
        // fftw_mpi_execute_dft_r2c(p_forward, wA, wAs);
        fftw_execute_dft_r2c(p_forward, wA, wAs);

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            wAs[ijk][0] = wAs[ijk][0] / NxNyNz;
            wAs[ijk][1] = wAs[ijk][1] / NxNyNz;
        }
        // fftw_mpi_execute_dft_r2c(p_forward, wB, wBs);
        fftw_execute_dft_r2c(p_forward, wB, wBs);

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            wBs[ijk][0] = wBs[ijk][0] / NxNyNz;
            wBs[ijk][1] = wBs[ijk][1] / NxNyNz;
        }

        freeW0 = 0.0;
        freeW1 = 0.0;
        freeAB0 = 0.0;
        freeAB1 = 0.0;
        freeABW0 = 0.0;
        freeABW1 = 0.0;
        freeABW = 0.0;
        freeS = 0.0;

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            freeAB0 = freeAB0 + sum_factor[ijk] * hAB * u_k[ijk] * (phAs[ijk][0] * phBs[ijk][0] + phAs[ijk][1] * phBs[ijk][1]);
            freeAB1 = freeAB1 + sum_factor[ijk] * hAB * u_k[ijk] * (phAs[ijk][1] * phBs[ijk][0] - phAs[ijk][0] * phBs[ijk][1]);
            freeW0 = freeW0 - sum_factor[ijk] * ((wAs[ijk][0] * phAs[ijk][0] + wAs[ijk][1] * phAs[ijk][1]) +
                                                (wBs[ijk][0] * phBs[ijk][0] + wBs[ijk][1] * phBs[ijk][1]));
            freeW1 = freeW1 - sum_factor[ijk] * ((wAs[ijk][1] * phAs[ijk][0] - wAs[ijk][0] * phAs[ijk][1]) +
                                                (wBs[ijk][1] * phBs[ijk][0] - wBs[ijk][0] * phBs[ijk][1]));
        }

        // if (world_rank == 0)
        // {
        //     MPI_Reduce(MPI_IN_PLACE, &freeAB0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //     MPI_Reduce(MPI_IN_PLACE, &freeW0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //     MPI_Reduce(MPI_IN_PLACE, &freeAB1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //     MPI_Reduce(MPI_IN_PLACE, &freeW1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // }
        // else
        // {
        //     MPI_Reduce(&freeAB0, &freeAB0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //     MPI_Reduce(&freeW0, &freeW0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //     MPI_Reduce(&freeAB1, &freeAB1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //     MPI_Reduce(&freeW1, &freeW1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // }

        freeABW0 = freeAB0 + freeW0;
        freeABW1 = freeAB1 + freeW1;

        freeABW = freeABW0;
        freeS = -log(parfQ);

        freeOld = freeEnergy;
        freeEnergy = freeABW + freeS;
        freeDiff = fabs(freeEnergy - freeOld);

        // 10.Repeat until desired accuracy is reached//

        if (iter == 1 || iter % 50 == 0 || iter >= maxIter)
        {
            FILE *fp = fopen("ing.dat", "a");
            fprintf(fp, "%5d : %.8e, %.8e, err=%.4f\nstress1=%.6f, stress2=%.6f, stress3=%.6f\nlx=%.4f, ly=%.4f, lz=%.4f\n\n", iter, freeEnergy, inCompMax, err, stress[0], stress[1], stress[2], lx, ly, lz);
            fclose(fp);

            printf("%5d : %.8e, %.8e, err=%.4f\nstress1=%e, stress2=%e, stress3=%e\nlx=%e, ly=%e, lz=%e\n", iter, freeEnergy, inCompMax, err, stress[0], stress[1], stress[2], lx, ly, lz);

        }

        // end = MPI_Wtime();
        fp = fopen("freeABW", "a");
        fprintf(fp, "%d:\n", iter);
        fprintf(fp, "freeABWr=%.10lf\n", freeABW0);
        fprintf(fp, "freeABWs=%.10lf\n", freeABW1);
        // fprintf(fp, "time per iteration = %.7E\n", end - start);
        fclose(fp);


        // MPI_Barrier(MPI_COMM_WORLD);

        // MPI_Bcast(&freeDiff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } while (iter < maxIter && (inCompMax > Sm1 || freeDiff > Sm2 || err > Sm3 || fabs(stress[0]) > Sm4 || fabs(stress[1]) > Sm4 || fabs(stress[2]) > Sm4));


    fp = fopen("ing.dat", "a");
    fprintf(fp, "%5d : %.8e, %.8e, err=%.4f\nstress1=%.6f, stress2=%.6f, stress3=%.6f\nlx=%.4f, ly=%.4f, lz=%.4f\n\n", iter, freeEnergy, inCompMax, err, stress[0], stress[1], stress[2], lx, ly, lz);
    fclose(fp);


    free(psuC);
    free(waDiff);
    free(wbDiff);

    fftw_free(phAs);
    fftw_free(phBs);

    fftw_free(wAs);
    fftw_free(wBs);
    free(wAnew);
    free(wBnew);
    fftw_free(wAnews);
    fftw_free(wBnews);

    free(eta);
    fftw_free(etas);

    free(del);
    free(outs);

    free(dudk);

    free(stress_FR_common);

    return freeEnergy;
}

//*****Calculate forward and backward propagators*****//
//*****and single chain pertition function*****//
//*****and then calculate PhA & PhB*****//
double getConc(double *phA, double *phB, double *wA, double *wB)
{
    int i, j, k, iz;
    long ijk, ijkiz;
    double *qA, *qcA, *qB, *qcB, *exp_wA, *exp_wB, *dgdk_A, *dgdk_B;
    double parfQ, *qInt;
    double *g_k;
    double *stress_bond_common;
    fftw_complex *qAs, *qcAs, *qBs, *qcBs;
    fftw_complex *qInts;

    g_k =    (double *)malloc(sizeof(double) * NxNyNz1);
    qA =     (double *)malloc(sizeof(double) * NxNyNz * NsA);
    qcA =    (double *)malloc(sizeof(double) * NxNyNz * NsA);
    qB =     (double *)malloc(sizeof(double) * NxNyNz * NsB);
    qcB =    (double *)malloc(sizeof(double) * NxNyNz * NsB);
    qInt =   (double *)malloc(sizeof(double) * NxNyNz);
    exp_wA = (double *)malloc(sizeof(double) * NxNyNz);
    exp_wB = (double *)malloc(sizeof(double) * NxNyNz);
    stress_bond_common = (double *)malloc(sizeof(double) * NxNyNz1);

    qAs =   (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1 * NsA);
    qcAs =  (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1 * NsA);
    qBs =   (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1 * NsB);
    qcBs =  (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1 * NsB);
    qInts = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * NxNyNz1);

    dgdk_A = (double *)malloc(sizeof(double) * NxNyNz1);
    dgdk_B = (double *)malloc(sizeof(double) * NxNyNz1);

    // Calculate all e^-w//
    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        exp_wA[ijk] = exp(-wA[ijk] / ((double)(N)));
    }
    // for (k = 0; k < Nz; k++)
    //     for (j = 0; j < Ny; j++)
    //         for (i = Nx; i < (Nx); i++)
    //         {
    //             ijk = (k * Ny + j) * (Nx) + i;
    //             exp_wA[ijk] = 0.0;
    //         }

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        exp_wB[ijk] = exp(-wB[ijk] / ((double)(N)));
    }
    // for (k = 0; k < Nz; k++)
    //     for (j = 0; j < Ny; j++)
    //         for (i = Nx; i < (Nx); i++)
    //         {
    //             ijk = (k * Ny + j) * (Nx) + i;
    //             exp_wB[ijk] = 0.0;
    //         }

    // Calculation for AB diblock copolymer//
    get_q(qA, qAs, exp_wA, blA, exp_wA, NsA, 1); // 0 to fA for qA

    for (ijk = 0; ijk < NxNyNz1; ijk++)
    {
        qInts[ijk][0] = qAs[(NsA - 1) * NxNyNz1 + ijk][0];
        qInts[ijk][1] = qAs[(NsA - 1) * NxNyNz1 + ijk][1];
    }

    conjunction_propagation(qInt, qInts, exp_wB, blA); // A->B

    get_q(qB, qBs, exp_wB, blB, qInt, NsB, 1); // fA to 1 for qB

    get_q(qcB, qcBs, exp_wB, blB, exp_wB, NsB, -1); // 1 to fA for qcB

    for (ijk = 0; ijk < NxNyNz1; ijk++)
    {
        qInts[ijk][0] = qcBs[ijk][0];
        qInts[ijk][1] = qcBs[ijk][1];
    }

    conjunction_propagation(qInt, qInts, exp_wA, blA); // A<-B

    get_q(qcA, qcAs, exp_wA, blA, qInt, NsA, -1); // fa to 0 for qcA

    // Calculate the single chain patition function//
    parfQ = 0.0;
    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        parfQ += qB[(NsB - 1) * NxNyNz + ijk];
    }
    // MPI_Allreduce(MPI_IN_PLACE, &parfQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    parfQ /= NxNyNz;

    // //Test the single chain partition function of diblock copolymer//
    //         for(iz=0; iz<NsA; iz++)
    //         {
    //                 parfQ=0.0;
    //                 for(k=0;k<local_Nz;k++)for(j=0;j<Ny;j++)for(i=0;i<Nx;i++)
    //                 {
    //                         ijk=(k*Ny+j)*(Nx+2)+i;
    //                                parfQ+=(qA[iz*local_Nx_2_NyNz+ijk]*qcA[iz*local_Nx_2_NyNz+ijk]/exp_wA[ijk]);
    //                 }
    //                 MPI_Allreduce(MPI_IN_PLACE, &parfQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //                 parfQ/=NxNyNz;
    //                 if(world_rank == 0)
    //                 {
    //                         printf("%lf\n",parfQ);
    //                 }
    //         }
    //         if(world_rank == 0)
    //         {
    //                 printf("*********\n");
    //         }
    //         for(iz=0; iz<NsB; iz++)
    //         {
    //                 parfQ=0.0;
    //                 for(k=0;k<local_Nz;k++)for(j=0;j<Ny;j++)for(i=0;i<Nx;i++)
    //                 {
    //                         ijk=(k*Ny+j)*(Nx+2)+i;
    //                                parfQ+=(qB[iz*local_Nx_2_NyNz+ijk]*qcB[iz*local_Nx_2_NyNz+ijk]/exp_wB[ijk]);
    //                 }
    //                 MPI_Allreduce(MPI_IN_PLACE, &parfQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //                 parfQ/=NxNyNz;
    //                 if(world_rank == 0)
    //                 {
    //                         printf("%lf\n",parfQ);
    //                 }
    //         }
    //         if(world_rank == 0)
    //         {
    //                 printf("*********\n");
    //         }

    // Calculate phA and phB//
    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (k * Ny + j) * (Nx) + i;

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;

                for (iz = 0; iz < NsA; iz++)
                {
                    phA[ijk] += (qA[iz * NxNyNz + ijk] * qcA[iz * NxNyNz + ijk] / exp_wA[ijk]);
                }

                for (iz = 0; iz < NsB; iz++)
                {
                    phB[ijk] += (qB[iz * NxNyNz + ijk] * qcB[iz * NxNyNz + ijk] / exp_wB[ijk]);
                }
                phA[ijk] *= parfQ * N;
                phB[ijk] *= parfQ * N;
            }

    // for (k = 0; k < Nz; k++)
    //     for (j = 0; j < Ny; j++)
    //         for (i = Nx; i < (Nx); i++)
    //         {
    //             ijk = (k * Ny + j) * (Nx) + i;

    //             phA[ijk] = 0.0;
    //             phB[ijk] = 0.0;
    //         }

    // //Test the calculation of phi//
    // double average_A=0.0, average_B=0.0;
    // for(ijk=0; ijk<local_Nx_2_NyNz; ijk++)
    // {
    //         average_A+=phA[ijk];
    //         average_B+=phB[ijk];
    // }
    // MPI_Allreduce(MPI_IN_PLACE, &average_A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE, &average_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // average_A/=NxNyNz;
    // average_B/=NxNyNz;
    // if(world_rank==0)
    // {
    //         printf("average = %lf\t%lf\t%lf\n", average_A, average_B);
    // }

    // Stress//
    // if (world_rank == 0)
    // {
        for (ijk = 1; ijk < NxNyNz1; ijk++)
        {
            dgdk_A[ijk] = (blA / (2 * kxyz[ijk])) * (kxyz[ijk] * blA * cos(kxyz[ijk] * blA) - sin(kxyz[ijk] * blA)) / (kxyz[ijk] * kxyz[ijk] * blA * blA);
        }
        dgdk_A[0] = 0.0;
    // }
    // else
    // {
    //     for (ijk = 0; ijk < local_NxNyNz1; ijk++)
    //     {
    //         dgdk_A[ijk] = (blA / (2 * kxyz[ijk])) * (kxyz[ijk] * blA * cos(kxyz[ijk] * blA) - sin(kxyz[ijk] * blA)) / (kxyz[ijk] * kxyz[ijk] * blA * blA);
    //     }
    // }

    // if (world_rank == 0)
    // {
        for (ijk = 1; ijk < NxNyNz1; ijk++)
        {
            dgdk_B[ijk] = (blB / (2 * kxyz[ijk])) * (kxyz[ijk] * blB * cos(kxyz[ijk] * blB) - sin(kxyz[ijk] * blB)) / (kxyz[ijk] * kxyz[ijk] * blB * blB);
        }
        dgdk_B[0] = 0.0;
    // }
    // else
    // {
    //     for (ijk = 0; ijk < local_NxNyNz1; ijk++)
    //     {
    //         dgdk_B[ijk] = (blB / (2 * kxyz[ijk])) * (kxyz[ijk] * blB * cos(kxyz[ijk] * blB) - sin(kxyz[ijk] * blB)) / (kxyz[ijk] * kxyz[ijk] * blB * blB);
    //     }
    // }

    for (ijk = 0; ijk < NxNyNz1; ijk++)
    {
        stress_bond_common[ijk] = 0.0;

        for (iz = 0; iz < NsA - 1; iz++)
        {
            stress_bond_common[ijk] += (1.0 / parfQ) * dgdk_A[ijk] * (qAs[iz * NxNyNz1 + ijk][0] * qcAs[(iz + 1) * NxNyNz1 + ijk][0] + qAs[iz * NxNyNz1 + ijk][1] * qcAs[(iz + 1) * NxNyNz1 + ijk][1]);
        }
        stress_bond_common[ijk] += (1.0 / parfQ) * dgdk_A[ijk] * (qAs[(NsA - 1) * NxNyNz1 + ijk][0] * qcBs[ijk][0] + qAs[(NsA - 1) * NxNyNz1 + ijk][1] * qcBs[ijk][1]);
        for (iz = 0; iz < NsB - 1; iz++)
        {
            stress_bond_common[ijk] += (1.0 / parfQ) * dgdk_B[ijk] * (qBs[iz * NxNyNz1 + ijk][0] * qcBs[(iz + 1) * NxNyNz1 + ijk][0] + qBs[iz * NxNyNz1 + ijk][1] * qcBs[(iz + 1) * NxNyNz1 + ijk][1]);
        }
    }

    stress_bond[0] = 0.0;
    for (ijk = 0; ijk < NxNyNz1; ijk++)
    {
        stress_bond[0] += sum_factor[ijk] * (-2 / lx) * kx_3d[ijk] * stress_bond_common[ijk];
    }

    stress_bond[1] = 0.0;
    for (ijk = 0; ijk < NxNyNz1; ijk++)
    {
        stress_bond[1] += sum_factor[ijk] * (-2 / ly) * ky_3d[ijk] * stress_bond_common[ijk];
    }

    stress_bond[2] = 0.0;
    for (ijk = 0; ijk < NxNyNz1; ijk++)
    {
        stress_bond[2] += sum_factor[ijk] * (-2 / lz) * kz_3d[ijk] * stress_bond_common[ijk];
    }

    free(g_k);
    free(qA);
    free(qB);
    free(qcA);
    free(qcB);
    free(qInt);
    free(exp_wA);
    free(exp_wB);
    free(stress_bond_common);

    fftw_free(qAs);
    fftw_free(qcAs);
    fftw_free(qBs);
    fftw_free(qcBs);
    fftw_free(qInts);

    free(dgdk_A);
    free(dgdk_B);

    return parfQ;
}

//*****Calculate propagator*****//
void get_q(double *prop, fftw_complex *props, double *exp_w, double bl, double *qInt, int ns, int sign)
{
    int i, j, k, iz;
    unsigned long ijk;
    double *g_k;

    g_k = (double *)malloc(sizeof(double) * NxNyNz1);

    // Define the ransition probability//
    // if (world_rank == 0)
    // {
        for (ijk = 1; ijk < NxNyNz1; ijk++)
        {
            g_k[ijk] = ((sin(kxyz[ijk] * bl)) / (kxyz[ijk] * bl)) / NxNyNz;
        }
        g_k[0] = 1.0 / NxNyNz;
    // }
    // else
    // {
    //     for (ijk = 0; ijk < NxNyNz1; ijk++)
    //     {
    //         g_k[ijk] = ((sin(kxyz[ijk] * bl)) / (kxyz[ijk] * bl)) / NxNyNz;
    //     }
    // }

    if (sign == 1)
    {
        for (ijk = 0; ijk < NxNyNz; ijk++)
        {
            prop[ijk] = qInt[ijk];
        }

        for (iz = 1; iz < ns; iz++)
        {
            // fftw_mpi_execute_dft_r2c(p_forward, prop, out);
            fftw_execute_dft_r2c(p_forward, prop, out);

            for (ijk = 0; ijk < NxNyNz1; ijk++)
            {
                props[ijk][0] = out[ijk][0] / NxNyNz;
                props[ijk][1] = out[ijk][1] / NxNyNz;
            }

            props += NxNyNz1;

            for (ijk = 0; ijk < NxNyNz1; ijk++)
            {
                out[ijk][0] *= g_k[ijk];
                out[ijk][1] *= g_k[ijk];
            }

            prop += NxNyNz;

            // fftw_mpi_execute_dft_c2r(p_backward, out, prop);
            fftw_execute_dft_c2r(p_backward, out, prop);

            for (ijk = 0; ijk < NxNyNz; ijk++)
            {
                prop[ijk] = prop[ijk] * exp_w[ijk];
            }
        }

        // fftw_mpi_execute_dft_r2c(p_forward, prop, out);
        fftw_execute_dft_r2c(p_forward, prop, out);

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            props[ijk][0] = out[ijk][0] / NxNyNz;
            props[ijk][1] = out[ijk][1] / NxNyNz;
        }

        prop -= NxNyNz * (ns - 1);
        props -= NxNyNz1 * (ns - 1);
    }
    else
    {
        prop += NxNyNz * (ns - 1);
        props += NxNyNz1 * (ns - 1);

        for (ijk = 0; ijk < NxNyNz; ijk++)
        {
            prop[ijk] = qInt[ijk];
        }

        for (iz = 1; iz < ns; iz++)
        {
            // fftw_mpi_execute_dft_r2c(p_forward, prop, out);
            fftw_execute_dft_r2c(p_forward, prop, out);

            for (ijk = 0; ijk < NxNyNz1; ijk++)
            {
                props[ijk][0] = out[ijk][0] / NxNyNz;
                props[ijk][1] = out[ijk][1] / NxNyNz;
            }

            props -= NxNyNz1;

            for (ijk = 0; ijk < NxNyNz1; ijk++)
            {
                out[ijk][0] *= g_k[ijk];
                out[ijk][1] *= g_k[ijk];
            }

            prop -= NxNyNz;

            // fftw_mpi_execute_dft_c2r(p_backward, out, prop);
            fftw_execute_dft_c2r(p_backward, out, prop);

            for (ijk = 0; ijk < NxNyNz; ijk++)
            {
                prop[ijk] = prop[ijk] * exp_w[ijk];
            }
        }

        // fftw_mpi_execute_dft_r2c(p_forward, prop, out);
        fftw_execute_dft_r2c(p_forward, prop, out);

        for (ijk = 0; ijk < NxNyNz1; ijk++)
        {
            props[ijk][0] = out[ijk][0] / NxNyNz;
            props[ijk][1] = out[ijk][1] / NxNyNz;
        }
    }
    free(g_k);
}

void conjunction_propagation(double *qInt, fftw_complex *qInts, double *exp_w, double bl)
{
    int i, j, k;
    long ijk;
    double *g_k;

    g_k = (double *)malloc(sizeof(double) * NxNyNz1);

    // Calculate the transition propability of the conjunction bond//
    // if (world_rank == 0)
    // {
        for (ijk = 1; ijk < NxNyNz1; ijk++)
        {
            g_k[ijk] = ((sin(kxyz[ijk] * bl)) / (kxyz[ijk] * bl));
        }
        g_k[0] = 1.0;
    // }
    // else
    // {
    //     for (ijk = 0; ijk < local_NxNyNz1; ijk++)
    //     {
    //         g_k[ijk] = ((sin(kxyz[ijk] * bl)) / (kxyz[ijk] * bl));
    //     }
    // }

    for (ijk = 0; ijk < NxNyNz1; ijk++)
    {
        qInts[ijk][0] *= g_k[ijk];
        qInts[ijk][1] *= g_k[ijk];
    }

    // fftw_mpi_execute_dft_c2r(p_backward, qInts, qInt);
    fftw_execute_dft_c2r(p_backward, qInts, qInt);

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        qInt[ijk] = qInt[ijk] * exp_w[ijk];
    }

    free(g_k);
}

// Calculate the error between in and out w field//
double error_cal(double *waDiffs, double *wbDiffs, double *wAs, double *wBs)
{
    double err_dif, err_w, err;
    int ijk;

    err = 0.0;
    err_dif = 0.0;
    err_w = 0.0;

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        err_dif += pow(waDiffs[ijk], 2) + pow(wbDiffs[ijk], 2);
        err_w += pow(wAs[ijk], 2) + pow(wBs[ijk], 2);
    }

    // if (world_rank == 0)
    // {
    //     MPI_Reduce(MPI_IN_PLACE, &err_dif, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //     MPI_Reduce(MPI_IN_PLACE, &err_w, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // }
    // else
    // {
    //     MPI_Reduce(&err_dif, &err_dif, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //     MPI_Reduce(&err_w, &err_w, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // }

    err = err_dif / err_w;
    err = sqrt(err);

    return err;
}

// Update the record of data of preceding N_hist iterations used in Anderson mixing//
void update_flds_hist(double *waDiff, double *wbDiff, double *wAnew, double *wBnew, double *del, double *outs)
{
    int ijk, j;

    del += hist_Nx * (N_hist - 1);
    outs += hist_Nx * (N_hist - 1);

    for (j = 0; j < N_hist - 1; j++)
    {

        for (ijk = 0; ijk < hist_Nx; ijk++)
        {
            del[ijk] = del[ijk - hist_Nx];
            outs[ijk] = outs[ijk - hist_Nx];
        }
        del -= hist_Nx;
        outs -= hist_Nx;
    }

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        del[ijk] = waDiff[ijk];
        outs[ijk] = wAnew[ijk];
    }

    del += NxNyNz;
    outs += NxNyNz;

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        del[ijk] = wbDiff[ijk];
        outs[ijk] = wBnew[ijk];
    }

    del += NxNyNz;
    outs += NxNyNz;

    del[0] = stress[0];
    del[1] = stress[1];
    del[2] = stress[2];
    outs[0] = lattice_para_new[0];
    outs[1] = lattice_para_new[1];
    outs[2] = lattice_para_new[2];

    del -= 2 * NxNyNz;
    outs -= 2 * NxNyNz;
}

/*********************************************************************/
/*
    Anderson mixing [O(Nx)]

    CHECKED
*/

void Anderson_mixing(double *del, double *outs, int N_rec, double *wA, double *wB)
{
    int i, k, ijk;
    int n, m;
    double *U, *V, *A, temp;
    int s;

    gsl_matrix_view uGnu;
    gsl_vector_view vGnu, aGnu;
    gsl_permutation *p;

    U = (double *)malloc(sizeof(double) * (N_rec - 1) * (N_rec - 1));
    V = (double *)malloc(sizeof(double) * (N_rec - 1));
    A = (double *)malloc(sizeof(double) * (N_rec - 1));

    /*
        Calculate the U-matrix and the V-vector
        Follow Shuang, and add the A and B components together.
    */

    for (n = 0; n < N_rec - 1; n++)
    {
        temp = 0.0;

        for (ijk = 0; ijk < hist_Nx; ijk++)
        {
            temp += (del[ijk] - del[(n + 1) * hist_Nx + ijk]) * del[ijk];
        }
        // if (world_rank == 0)
        // {
        //     MPI_Reduce(MPI_IN_PLACE, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // }
        // else
        // {
        //     MPI_Reduce(&temp, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // }

        V[n] = temp;

        for (m = n; m < N_rec - 1; m++)
        {
            temp = 0.0;

            for (ijk = 0; ijk < hist_Nx; ijk++)
            {
                temp += (del[ijk] - del[(n + 1) * hist_Nx + ijk]) * (del[ijk] - del[(m + 1) * hist_Nx + ijk]);
            }
            // if (world_rank == 0)
            // {
            //     MPI_Reduce(MPI_IN_PLACE, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // }
            // else
            // {
            //     MPI_Reduce(&temp, &temp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // }

            U[(N_rec - 1) * n + m] = temp;
            U[(N_rec - 1) * m + n] = U[(N_rec - 1) * n + m];
        }
    }

    /* Calculate A - uses GNU LU decomposition for U A = V */
    // if (world_rank == 0)
    // {
        uGnu = gsl_matrix_view_array(U, N_rec - 1, N_rec - 1);
        vGnu = gsl_vector_view_array(V, N_rec - 1);
        aGnu = gsl_vector_view_array(A, N_rec - 1);

        p = gsl_permutation_alloc(N_rec - 1);

        gsl_linalg_LU_decomp(&uGnu.matrix, p, &s);

        gsl_linalg_LU_solve(&uGnu.matrix, p, &vGnu.vector, &aGnu.vector);

        gsl_permutation_free(p);
    // }

    // MPI_Bcast(A, N_rec - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Update omega */

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        wA[ijk] = outs[ijk];
    }

    outs += NxNyNz;

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        wB[ijk] = outs[ijk];
    }

    outs += NxNyNz;

    lattice_para[0] = outs[0];
    lattice_para[1] = outs[1];
    lattice_para[2] = outs[2];

    outs -= NxNyNz * 2;

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        for (n = 0; n < N_rec - 1; n++)
        {
            wA[ijk] += A[n] * (outs[(n + 1) * hist_Nx + ijk] - outs[ijk]);
        }
    }

    outs += NxNyNz;

    for (ijk = 0; ijk < NxNyNz; ijk++)
    {
        for (n = 0; n < N_rec - 1; n++)
        {
            wB[ijk] += A[n] * (outs[(n + 1) * hist_Nx + ijk] - outs[ijk]);
        }
    }

    outs += NxNyNz;

    for (n = 0; n < N_rec - 1; n++)
    {
        lattice_para[0] += A[n] * (outs[(n + 1) * hist_Nx] - outs[0]);
        lattice_para[1] += A[n] * (outs[(n + 1) * hist_Nx + 1] - outs[1]);
        lattice_para[2] += A[n] * (outs[(n + 1) * hist_Nx + 2] - outs[2]);
    }

    outs -= NxNyNz * 2;

    free(A);
    free(V);
    free(U);
}

// Initialization//
void init_reading(double *wA, double *wB, double *phA, double *phB)
{
    int i, j, k;
    long ijk;
    FILE *fp;
    fp = fopen("phiin.dat", "r");
    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                fscanf(fp, "%lf %lf %lf %lf\n", &phA[ijk], &phB[ijk], &wA[ijk], &wB[ijk]);
            }
        }
    }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_Disordered(double *wA, double *wB, double *phA, double *phB)
{
    int i, j, k;
    long ijk;
    FILE *fp = fopen("init_phA.dat", "w");

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = f;
                phB[ijk] = fB;

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_Lamella(double *wA, double *wB, double *phA, double *phB)
{
    int i, j, k;
    long ijk;
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(lx * f / 2.0, 2.0);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;
                rx0 = 0.5 * lx;
                rsqd = (rx - rx0) * (rx - rx0);
                if (rsqd <= r0sqd)
                {
                    phA[ijk] = 1.0;
                    phB[ijk] = 0.0;
                }
                else
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }
                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_HEX(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[5][2] = {{0.5, 0.5},
                        {0.0, 0.0},
                        {1.0, 0.0},
                        {0.0, 1.0},
                        {1.0, 1.0}};
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = 0.25 * lx * ly * 0.3 / Pi;

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 5; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_BCC(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[9][3] = {{0.5, 0.5, 0.5},
                         {0, 0, 0},
                         {1, 0, 0},
                         {0, 1, 0},
                         {1, 1, 0},
                         {0, 0, 1},
                         {1, 0, 1},
                         {0, 1, 1},
                         {1, 1, 1}};

    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.375 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 9; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_FCC(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[14][3] = {{0.5, 0.0, 0.5},
                        {0.0, 0.5, 0.5},
                        {0.5, 1.0, 0.5},
                        {1.0, 0.5, 0.5},
                        {0.5, 0.5, 0.0},
                        {0.5, 0.5, 1.0},
                        {0, 0, 0},
                        {1, 0, 1},
                        {0, 1, 0},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 0, 0},
                        {0, 1, 1},
                        {1, 1, 1}};

    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.125 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 14; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_HCP(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[13][3] = {{0.5, 0.1667, 0.5},
                        {0.0, 0.6667, 0.5},
                        {1, 0.6667, 0.5},
                        {0, 0, 0},
                        {1, 0, 1},
                        {0, 1, 0},
                        {0, 0, 1},
                        {1, 1, 0},
                        {1, 0, 0},
                        {0, 1, 1},
                        {1, 1, 1},
                        {0.5, 0.5, 0},
                        {0.5, 0.5, 1}};

    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.2 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 13; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_C14(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[41][3] = {{0, 0, 0},
                        {0, 0, 1},
                        {0, 1, 0},
                        {0, 1, 1},
                        {1, 1, 0},
                        {1, 1, 1},
                        {1, 0, 0},
                        {1, 0, 1},
                        {0.5, 0.5, 0.5},
                        {0.5, 0.5, 0},
                        {0.5, 0.5, 1},
                        {1, 1, 0.5},
                        {1, 0, 0.5},
                        {0, 0, 0.5},
                        {0, 1, 0.5},
                        {0.74575, 0.91525, 0.75},
                        {0.25425, 0.91525, 0.75},
                        {1, 0.8305, 0.25},
                        {0, 0.8305, 0.25},
                        {0.5, 0.6695, 0.75},
                        {0.75425, 0.58475, 0.25},
                        {0.24575, 0.58475, 0.25},
                        {0.75425, 0.41525, 0.75},
                        {0.24575, 0.41525, 0.75},
                        {0.5, 0.3305, 0.25},
                        {1, 0.1695, 0.75},
                        {0.74575, 0.08475, 0.25},
                        {0.25425, 0.08475, 0.25},
                        {0, 0.1695, 0.75},
                        {0.5, 0.83334, 0.438},
                        {0.5, 0.83334, 0.062},
                        {1, 0.66666, 0.562},
                        {1, 0.66666, 0.938},
                        {0, 0.66666, 0.562},
                        {0, 0.66666, 0.938},
                        {1, 0.33334, 0.062},
                        {1, 0.33334, 0.438},
                        {0.5, 0.16666, 0.938},
                        {0.5, 0.16666, 0.562},
                        {0, 0.33334, 0.438},
                        {0, 0.33334, 0.062}};

    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.03 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 41; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_C15(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[34][3] = {{0.87500, 0.37500, 0.12500},
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

    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.03 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 34; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_A15(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[21][3] = {{0, 0, 0},
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

    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.09 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 21; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_Sigma(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[47][3] = {{0, 0, 0},
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

    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.03 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 47; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_Z(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int i, j, k, m, counter;
    long ijk;
    double temp;
    double Coor[32][3] = {{0.5, 0.0, 1.0},
                        {0.5, 0.0, 0.0},
                        {0.75, 0.25, 1.0},
                        {0.75, 0.25, 0.0},
                        {0.25, 0.25, 1.0},
                        {0.25, 0.25, 0.0},
                        {1.0, 0.5, 1.0},
                        {1.0, 0.5, 0.0},
                        {0.0, 0.5, 1.0},
                        {0.0, 0.5, 0.0},
                        {0.75, 0.75, 1.0},
                        {0.75, 0.75, 0.0},
                        {0.25, 0.75, 1.0},
                        {0.25, 0.75, 0.0},
                        {0.5, 1.0, 1.0},
                        {0.5, 1.0, 0.0},
                        {0.0, 0.0, 0.25},
                        {1.0, 0.0, 0.25},
                        {0.0, 1.0, 0.25},
                        {1.0, 1.0, 0.25},
                        {0.5, 0.5, 0.25},
                        {0.0, 0.0, 0.75},
                        {1.0, 0.0, 0.75},
                        {0.0, 1.0, 0.75},
                        {1.0, 1.0, 0.75},
                        {0.5, 0.5, 0.75},
                        {0.5, 0.8333, 0.5},
                        {1.0, 0.6667, 0.5},
                        {0.0, 0.6667, 0.5},
                        {1.0, 0.3333, 0.5},
                        {0.0, 0.3333, 0.5},
                        {0.5, 0.1667, 0.5}};
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = pow(0.02 * lx * ly * lz * 0.25 / Pi, 2.0 / 3);

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                rx = (i + 0.5) * dx;
                ry = (j + 0.5) * dy;
                rz = (k + 0.5) * dz;

                counter = 0;
                for (m = 0; m < 32; m++)
                {
                    rx0 = Coor[m][0] * lx;
                    ry0 = Coor[m][1] * ly;
                    rz0 = Coor[m][2] * lz;
                    rsqd = (rx - rx0) * (rx - rx0) + (ry - ry0) * (ry - ry0) + (rz - rz0) * (rz - rz0);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        phB[ijk] = 0.0;

                        counter = 1;
                        break;
                    }
                }

                if (counter == 0)
                {
                    phA[ijk] = 0.0;
                    phB[ijk] = 1.0;
                }

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_Gyroid(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int count_A;
    double target_ratio, ratio_A, diff_A, epsl;
    int i, j, k;
    long ijk;
    double temp;
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = -0.5;
    count_A = 0;

    epsl = 1.0;

    target_ratio = 0.4;

    while (epsl > 0.0005)
    {
        count_A = 0;

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nx; i++)
                {
                    ijk = (long)((k * Ny + j) * (Nx) + i);
                    rx = (i + 0.5) * dx * 2.0 * Pi / lx;
                    ry = (j + 0.5) * dy * 2.0 * Pi / ly;
                    rz = (k + 0.5) * dz * 2.0 * Pi / lz;

                    rsqd = sin(rx) * cos(ry) + sin(ry) * cos(rz) + sin(rz) * cos(rx);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        count_A += 1;
                    }
                    else
                    {
                        phA[ijk] = 0.0;
                    }
                }

        ratio_A = 1.0 * count_A / NxNyNz;
        if (ratio_A >= target_ratio)
        {
            epsl = ratio_A - target_ratio;
            r0sqd -= 0.001;
        }
        else
        {
            epsl = target_ratio - ratio_A;
            r0sqd += 0.001;
        }
    }

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phB[ijk] = 1.0 - phA[ijk];

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_DoubleGyroid(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int count_A;
    double target_ratio, ratio_A, diff_A, epsl;
    int i, j, k;
    long ijk;
    double temp;
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = -0.5;
    count_A = 0;

    epsl = 1.0;

    target_ratio = 0.4;

    while (epsl > 0.0005)
    {
        count_A = 0;

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nx; i++)
                {
                    ijk = (long)((k * Ny + j) * (Nx) + i);
                    rx = (i + 0.5) * dx * 2.0 * Pi / lx;
                    ry = (j + 0.5) * dy * 2.0 * Pi / ly;
                    rz = (k + 0.5) * dz * 2.0 * Pi / lz;

                    rsqd = sin(rx) * cos(ry) + sin(ry) * cos(rz) + sin(rz) * cos(rx);
                    if (rsqd <= r0sqd || rsqd >= -r0sqd)
                    {
                        phA[ijk] = 1.0;
                        count_A += 1;
                    }
                    else
                    {
                        phA[ijk] = 0.0;
                    }
                }

        ratio_A = 1.0 * count_A / NxNyNz;
        if (ratio_A >= target_ratio)
        {
            epsl = ratio_A - target_ratio;
            r0sqd -= 0.001;
        }
        else
        {
            epsl = target_ratio - ratio_A;
            r0sqd += 0.001;
        }
    }

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phB[ijk] = 1.0 - phA[ijk];

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_Diamond(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int count_A;
    double target_ratio, ratio_A, diff_A, epsl;
    int i, j, k;
    long ijk;
    double temp;
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = -0.5;
    count_A = 0;

    epsl = 1.0;

    target_ratio = 0.4;

    while (epsl > 0.0005)
    {
        count_A = 0;

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nx; i++)
                {
                    ijk = (long)((k * Ny + j) * (Nx) + i);
                    rx = (i + 0.5) * dx * 2.0 * Pi / lx;
                    ry = (j + 0.5) * dy * 2.0 * Pi / ly;
                    rz = (k + 0.5) * dz * 2.0 * Pi / lz;

                    rsqd = cos(rx) * cos(ry) * cos(rz) + sin(rx) * sin(ry) * sin(rz);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        count_A += 1;
                    }
                    else
                    {
                        phA[ijk] = 0.0;
                    }
                }

        ratio_A = 1.0 * count_A / NxNyNz;
        if (ratio_A >= target_ratio)
        {
            epsl = ratio_A - target_ratio;
            r0sqd -= 0.001;
        }
        else
        {
            epsl = target_ratio - ratio_A;
            r0sqd += 0.001;
        }
    }

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phB[ijk] = 1.0 - phA[ijk];

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_DoubleDiamond(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int count_A;
    double target_ratio, ratio_A, diff_A, epsl;
    int i, j, k;
    long ijk;
    double temp;
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = -0.5;
    count_A = 0;

    epsl = 1.0;

    target_ratio = 0.6;

    while (epsl > 0.0005)
    {
        count_A = 0;

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nx; i++)
                {
                    ijk = (long)((k * Ny + j) * (Nx) + i);
                    rx = (i + 0.5) * dx * 2.0 * Pi / lx;
                    ry = (j + 0.5) * dy * 2.0 * Pi / ly;
                    rz = (k + 0.5) * dz * 2.0 * Pi / lz;

                    rsqd = cos(rx) * cos(ry) * cos(rz) + sin(rx) * sin(ry) * sin(rz);
                    if (rsqd <= r0sqd || rsqd >= -r0sqd)
                    {
                        phA[ijk] = 1.0;
                        count_A += 1;
                    }
                    else
                    {
                        phA[ijk] = 0.0;
                    }
                }

        ratio_A = 1.0 * count_A / NxNyNz;
        if (ratio_A >= target_ratio)
        {
            epsl = ratio_A - target_ratio;
            r0sqd -= 0.001;
        }
        else
        {
            epsl = target_ratio - ratio_A;
            r0sqd += 0.001;
        }

        // printf("%lf, %lf\n", epsl, r0sqd);
    }

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phB[ijk] = 1.0 - phA[ijk];

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_O70(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int count_A;
    double target_ratio, ratio_A, diff_A, epsl;
    int i, j, k;
    long ijk;
    double temp;
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = -0.5;
    count_A = 0;

    epsl = 1.0;

    target_ratio = 0.4;

    while (epsl > 0.0005)
    {
        count_A = 0;

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nx; i++)
                {
                    ijk = (long)((k * Ny + j) * (Nx) + i);
                    rx = (i + 0.5) * dx * 2.0 * Pi / lx;
                    ry = (j + 0.5) * dy * 2.0 * Pi / ly;
                    rz = (k + 0.5) * dz * 2.0 * Pi / lz;

                    rsqd = cos(rx) * cos(ry) * cos(rz) + sin(rx) * sin(ry) * cos(rz) + sin(rx) * cos(ry) * sin(rz) + cos(rx) * sin(ry) * sin(rz);
                    if (rsqd <= r0sqd)
                    {
                        phA[ijk] = 1.0;
                        count_A += 1;
                    }
                    else
                    {
                        phA[ijk] = 0.0;
                    }
                }

        ratio_A = 1.0 * count_A / NxNyNz;
        if (ratio_A <= target_ratio)
        {
            epsl = ratio_A - target_ratio;
            r0sqd -= 0.001;
        }
        else
        {
            epsl = target_ratio - ratio_A;
            r0sqd += 0.001;
        }
    }

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phB[ijk] = 1.0 - phA[ijk];

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));

                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_DoublePlumberNightmare(double *wA, double *wB, double *phA, double *phB, int inv)
{
    int count_A;
    double target_ratio, ratio_A, diff_A, epsl;
    int i, j, k;
    long ijk;
    double temp;
    FILE *fp = fopen("init_phA.dat", "w");

    r0sqd = -0.8;
    count_A = 0;

    epsl = 1.0;

    target_ratio = 0.4;

    while (epsl > 0.0005 && r0sqd < -0.5 && r0sqd > -0.9)
    {
        count_A = 0;

        for (k = 0; k < Nz; k++)
            for (j = 0; j < Ny; j++)
                for (i = 0; i < Nx; i++)
                {
                    ijk = (long)((k * Ny + j) * (Nx) + i);
                    rx = (i + 0.5) * dx * 2.0 * Pi / lx;
                    ry = (j + 0.5) * dy * 2.0 * Pi / ly;
                    rz = (k + 0.5) * dz * 2.0 * Pi / lz;

                    rsqd = cos(rx) + cos(ry) + cos(rz);
                    if (rsqd <= r0sqd || rsqd >= -r0sqd)
                    {
                        phA[ijk] = 1.0;
                        count_A += 1;
                    }
                    else
                    {
                        phA[ijk] = 0.0;
                    }
                }

        ratio_A = 1.0 * count_A / NxNyNz;
        if (ratio_A <= target_ratio)
        {
            epsl = ratio_A - target_ratio;
            r0sqd -= 0.001;
        }
        else
        {
            epsl = target_ratio - ratio_A;
            r0sqd += 0.001;
        }
    }

    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);

                phB[ijk] = 1.0 - phA[ijk];

                if (inv == 0)
                {
                }
                else
                {
                    temp = phA[ijk];
                    phA[ijk] = phB[ijk];
                    phB[ijk] = temp;
                }

                wA[ijk] = hAB * phB[ijk] * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * phA[ijk] * (1.0 + 0.040 * (drand48() - 0.5));

                fprintf(fp, "%lf\n", phA[ijk]);
            }
    fclose(fp);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = Nx; i < (Nx); i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                phA[ijk] = 0.0;
                phB[ijk] = 0.0;
                wA[ijk] = 0.0;
                wB[ijk] = 0.0;
            }
        }
    }
}

void init_random_W(double *wA, double *wB)
{
    int i, j, k;
    long ijk;

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                wA[ijk] = hAB * fB * (1.0 + 0.040 * (drand48() - 0.5));
                wB[ijk] = hAB * f * (1.0 + 0.040 * (drand48() - 0.5));
            }
        }
    }
}

//********************Output configuration******************************
void write_ph(double *phA, double *phB, double *wA, double *wB)
{
    int i, j, k;
    long ijk;
    FILE *fp = fopen("phi.dat", "w");

    fprintf(fp, "%d %d %d\n%lf %lf %lf\n", Nz,Ny,Nx,lz,ly,lx);

    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                fprintf(fp, "%lf %lf %lf %lf\n", phA[ijk], phB[ijk], wA[ijk], wB[ijk]);
            }
        }
    }
    fclose(fp);
}

void write_phA(double *phA)
{
    int i, j, k;
    long ijk;
    FILE *fp = fopen("phA.dat", "w");
    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                fprintf(fp, "%lf\n", phA[ijk]);
            }
        }
    }
    fclose(fp);
}

void write_phB(double *phB)
{
    int i, j, k;
    long ijk;
    FILE *fp = fopen("phB.dat", "w");
    for (k = 0; k < Nz; k++)
    {
        for (j = 0; j < Ny; j++)
        {
            for (i = 0; i < Nx; i++)
            {
                ijk = (long)((k * Ny + j) * (Nx) + i);
                fprintf(fp, "%lf\n", phB[ijk]);
            }
        }
    }
    fclose(fp);
}

/*
void Distribute(double *wA, double *wB)
{
    int i, j, k;

    if (world_rank == 0)
    {
        for (i = 1; i < world_size; i++)
        {
            MPI_Send(&wA[i * local_Nx_2_NyNz], local_Nx_2_NyNz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&wB[i * local_Nx_2_NyNz], local_Nx_2_NyNz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(wA, local_Nx_2_NyNz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(wB, local_Nx_2_NyNz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void Gather(double *phA, double *phB, double *wA, double *wB)
{
    int i, j, k;

    if (world_rank == 0)
    {
        for (i = 1; i < world_size; i++)
        {
            MPI_Recv(&phA[i * local_Nx_2_NyNz], local_Nx_2_NyNz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&phB[i * local_Nx_2_NyNz], local_Nx_2_NyNz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&wA[i * local_Nx_2_NyNz], local_Nx_2_NyNz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&wB[i * local_Nx_2_NyNz], local_Nx_2_NyNz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        MPI_Send(phA, local_Nx_2_NyNz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(phB, local_Nx_2_NyNz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(wA, local_Nx_2_NyNz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(wB, local_Nx_2_NyNz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

*/