#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #include "mpi.h"
#include <fftw3.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#define MaxIT 200000  // Maximum iteration steps
#define Nx 240
#define Ny 240
// #define Nz 1			//grid size

#define NxNy (Nx * Ny)
#define Nyh1 (Ny / 2 + 1)
#define Ny2 (2 * Nyh1)
#define NxNy1 (Nx * Nyh1)

#define Pi 3.141592653589

void write_stress(double* stress_tensor);

void sovDifFft(double* g, double* w, double* qInt, double z, int ns, int sign);

int NsA, NsB;
int nsA, nsB;

int Narm;

double kx[Nx], ky[Ny], *kxyzdz, *wdz, dx, dy, dz;
double lx, ly, lz;

double hAB, fA, fB, ds0, ds2;

double*       in;
fftw_complex* out;
fftw_plan     p_forward, p_backward;

double temp;
char   FEname[50], phname[50];

int intag;

void allocate_mem_global(double* wA, double* wB, double* phA, double* phB);

void   init_w(double* wA, double* wB);
double tensor_calc(double* phlA,
                   double* phlB,
                   double* wA,
                   double* wB,
                   double* stress_tensor);

int main(int argc, char** argv)
{
    /* init variable */

    double *wA, *wB, *phA, *phB, *stress_tensor;

    allocate_mem_global(wA, wB, phA, phB);

    wA = (double*)malloc(sizeof(double) * NxNy);
    wB = (double*)malloc(sizeof(double) * NxNy);

    phA = (double*)malloc(sizeof(double) * NxNy);
    phB = (double*)malloc(sizeof(double) * NxNy);

    printf("successfully initialized parameter \n");

    init_w(wA, wB);

    //***************Initialize wA, wB******************
    printf("successfully initialized\n");

    stress_tensor = (double*)malloc(sizeof(double) * NxNy);

    tensor_calc(phA, phB, wA, wB, stress_tensor);

    write_stress(stress_tensor);

    free(stress_tensor);

    return 1;
}

double tensor_calc(double* phlA,
                   double* phlB,
                   double* wA,
                   double* wB,
                   double* stress_tensor)
{
    double ql, ffl, qtmp;
    long   ijk, ijkiz, ijk_1, ijk_2, ijk_3, ijk_4, ijk_5, ijk_6;
    int    m, iz, s;
    int    i, j, k;

    double* qA   = (double*)malloc(sizeof(double) * NxNy * (NsA + 1));
    double* qcA  = (double*)malloc(sizeof(double) * NxNy * (NsA + 1));
    double* qB   = (double*)malloc(sizeof(double) * NxNy * (NsB + 1));
    double* qcB  = (double*)malloc(sizeof(double) * NxNy * (NsB + 1));
    double* qInt = (double*)malloc(sizeof(double) * NxNy);
    double* fxA  = (double*)malloc(sizeof(double) * NxNy * (NsA + 1));
    double* fcxA = (double*)malloc(sizeof(double) * NxNy * (NsA + 1));
    double* fxB  = (double*)malloc(sizeof(double) * NxNy * (NsB + 1));
    double* fcxB = (double*)malloc(sizeof(double) * NxNy * (NsB + 1));

    double dx = lx / Nx;
    double dy = ly / Ny;

    for (ijk = 0; ijk < NxNy; ijk++) {
        qInt[ijk] = 1.0;
    }

    sovDifFft(qA, wA, qInt, fA, NsA, 1); /* A(n-1)+A_star_B */
    sovDifFft(qcB, wB, qInt, fB, NsB, -1);

    for (ijk = 0; ijk < NxNy; ijk++) {
        qInt[ijk] = qA[ijk * (NsA + 1) + NsA];
        qtmp      = qcB[ijk * (NsB + 1)];
        for (m = 1; m < Narm; m++)
            qInt[ijk] *= qtmp;
    }

    sovDifFft(qB, wB, qInt, fB, NsB, 1);  // fb to 1 for qB

    for (ijk = 0; ijk < NxNy; ijk++) {
        qInt[ijk] = qcB[ijk * (NsB + 1)];
        qtmp      = qcB[ijk * (NsB + 1)];
        for (m = 1; m < Narm; m++)
            qInt[ijk] *= qtmp;
    }

    sovDifFft(qcA, wA, qInt, fA, NsA, -1);  // fa to 0 for qcA

    ql = 0.0;
    for (ijk = 0; ijk < NxNy; ijk++) {
        ql += qB[ijk * (NsB + 1) + NsB];
    }

    ql /= NxNy;
    ffl = ds0 / ql;

    for (ijk = 0; ijk < NxNy; ijk++) {
        phlA[ijk] = 0.0;
        phlB[ijk] = 0.0;

        for (iz = 0; iz <= NsA; iz++) {
            ijkiz = ijk * (NsA + 1) + iz;
            if (iz == 0 || iz == NsA)
                phlA[ijk] += (0.50 * qA[ijkiz] * qcA[ijkiz]);
            else
                phlA[ijk] += (qA[ijkiz] * qcA[ijkiz]);
        }

        for (iz = 0; iz <= NsB; iz++) {
            ijkiz = ijk * (NsB + 1) + iz;
            if (iz == 0 || iz == NsB)
                phlB[ijk] += (0.50 * qB[ijkiz] * qcB[ijkiz]);
            else
                phlB[ijk] += (qB[ijkiz] * qcB[ijkiz]);
        }

        phlA[ijk] *= ffl;
        phlB[ijk] *= ffl;
    }

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++) {
            for (s = 0; s <= NsA; s++) {
                ijk   = i * Ny + j;
                ijk_1 = ((i + 1) % Nx) * Ny + j;
                ijk_2 = (i % Nx) * Ny + j;
                ijk_3 = ((i - 1 + Nx) % Nx) * Ny + j;

                ijk_4 = ((i) % Nx) * Ny + (j + 1) % Ny;
                ijk_5 = ((i) % Nx) * Ny + (j) % Ny;
                ijk_6 = ((i + Nx) % Nx) * Ny + (j - 1 + Ny) % Ny;

                fxA[ijk * (NsA + 1) + s] =
                    (qA[ijk_1 * (NsA + 1) + s] - qA[ijk_3 * (NsA + 1) + s]) /
                    (2 * dx);
                fcxA[ijk * (NsA + 1) + s] =
                    (qcA[ijk_1 * (NsA + 1) + s] - qcA[ijk_3 * (NsA + 1) + s]) /
                    (2 * dx);
                // stress_tensor[i+j*Nx+k*Nx*Ny]+=(qA[ijk_1*(NsA+1)+s]-qA[ijk_2*(NsA+1)+s])*(qcA[ijk_3*(NsA+1)+s]-qcA[ijk_4*(NsA+1)+s])*ds0/(4*dx*dy);
            }
            for (s = 0; s <= NsB; s++) {
                ijk   = i * Ny + j;
                ijk_1 = ((i + 1) % Nx) * Ny + j;
                ijk_2 = (i % Nx) * Ny + j;
                ijk_3 = ((i - 1 + Nx) % Nx) * Ny + j;

                ijk_4 = ((i) % Nx) * Ny + (j + 1) % Ny;
                ijk_5 = ((i) % Nx) * Ny + (j) % Ny;
                ijk_6 = ((i + Nx) % Nx) * Ny + (j - 1 + Ny) % Ny;

                fxB[ijk * (NsB + 1) + s] =
                    (qB[ijk_1 * (NsB + 1) + s] - qB[ijk_3 * (NsB + 1) + s]) /
                    (2 * dx);
                fcxB[ijk * (NsB + 1) + s] =
                    (qcB[ijk_1 * (NsB + 1) + s] - qcB[ijk_3 * (NsB + 1) + s]) /
                    (2 * dx);
                // stress_tensor[i+j*Nx+k*Nx*Ny]+=(qA[ijk_1*(NsA+1)+s]-qA[ijk_2*(NsA+1)+s])*(qcA[ijk_3*(NsA+1)+s]-qcA[ijk_4*(NsA+1)+s])*ds0/(4*dx*dy);
            }
        }
    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++) {
            stress_tensor[i * Ny + j] = 0;
            for (s = 0; s <= NsA; s++) {
                ijk   = i * Ny + j;
                ijk_1 = ((i + 1) % Nx) * Ny + j;
                ijk_2 = (i % Nx) * Ny + j;
                ijk_3 = ((i - 1 + Nx) % Nx) * Ny + j;

                ijk_4 = ((i) % Nx) * Ny + (j + 1) % Ny;
                ijk_5 = ((i) % Nx) * Ny + (j) % Ny;
                ijk_6 = ((i + Nx) % Nx) * Ny + (j - 1 + Ny) % Ny;

                if (s == 0 || s == NsA)
                    stress_tensor[i * Ny + j] +=
                        (0.5 * qA[ijk * (NsA + 1) + s] *
                             (fcxA[ijk_4 * (NsA + 1) + s] -
                              fcxA[ijk_6 * (NsA + 1) + s]) *
                             ds0 / (2 * dy) +
                         0.5 * qcA[ijk * (NsA + 1) + s] *
                             (fxA[ijk_4 * (NsA + 1) + s] -
                              fxA[ijk_6 * (NsA + 1) + s]) *
                             ds0 / (2 * dy));
                else
                    stress_tensor[i * Ny + j] +=
                        (qA[ijk * (NsA + 1) + s] *
                             (fcxA[ijk_4 * (NsA + 1) + s] -
                              fcxA[ijk_6 * (NsA + 1) + s]) *
                             ds0 / (2 * dy) +
                         qcA[ijk * (NsA + 1) + s] *
                             (fxA[ijk_4 * (NsA + 1) + s] -
                              fxA[ijk_6 * (NsA + 1) + s]) *
                             ds0 / (2 * dy));

                // stress_tensor[i+j*Nx+k*Nx*Ny]+=(qA[ijk_1*(NsA+1)+s]-qA[ijk_2*(NsA+1)+s])*(qcA[ijk_3*(NsA+1)+s]-qcA[ijk_4*(NsA+1)+s])*ds0/(4*dx*dy);
            }

            for (s = 0; s <= NsB; s++) {
                ijk   = i * Ny + j;
                ijk_1 = ((i + 1) % Nx) * Ny + j;
                ijk_2 = (i % Nx) * Ny + j;
                ijk_3 = ((i - 1 + Nx) % Nx) * Ny + j;

                ijk_4 = ((i) % Nx) * Ny + (j + 1) % Ny;
                ijk_5 = ((i) % Nx) * Ny + (j) % Ny;
                ijk_6 = ((i + Nx) % Nx) * Ny + (j - 1 + Ny) % Ny;

                if (s == 0 || s == NsB)
                    stress_tensor[i * Ny + j] +=
                        (0.5 * qB[ijk * (NsB + 1) + s] *
                             (fcxB[ijk_4 * (NsB + 1) + s] -
                              fcxB[ijk_6 * (NsB + 1) + s]) *
                             ds0 / (2 * dy) +
                         0.5 * qcB[ijk * (NsB + 1) + s] *
                             (fxB[ijk_4 * (NsB + 1) + s] -
                              fxB[ijk_6 * (NsB + 1) + s]) *
                             ds0 / (2 * dy));
                else
                    stress_tensor[i * Ny + j] +=
                        (qB[ijk * (NsB + 1) + s] *
                             (fcxB[ijk_4 * (NsB + 1) + s] -
                              fcxB[ijk_6 * (NsB + 1) + s]) *
                             ds0 / (2 * dy) +
                         qcB[ijk * (NsB + 1) + s] *
                             (fxB[ijk_4 * (NsB + 1) + s] -
                              fxB[ijk_6 * (NsB + 1) + s]) *
                             ds0 / (2 * dy));

                // stress_tensor[i+j*Nx+k*Nx*Ny]+=(qB[ijk_1*(NsB+1)+s]-qB[ijk_2*(NsB+1)+s])*(qcB[ijk_3*(NsB+1)+s]-qcB[ijk_4*(NsB+1)+s])*ds0/(4*dx*dy);
            }
        }

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
            stress_tensor[i * Ny + j] = stress_tensor[i * Ny + j] / ql;

    double stress = 0;

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
            stress += stress_tensor[i * Ny + j] * dx * dy;

    free(qA);
    free(qcA);
    free(qB);
    free(qcB);
    free(fxA);
    free(fcxA);
    free(fxB);
    free(fcxB);

    free(qInt);
    printf("%g\n", stress);
    printf("Nx=%d, Ny=%d\n", Nx, Ny);
    printf("dx=%lf, dy=%lf\n", dx, dy);
    return stress;
}

void sovDifFft(double* g, double* w, double* qInt, double z, int ns, int sign)
{
    int  i, j, k, iz, ns1;
    long ijk, ijkr;

    ns1 = ns + 1;

    for (ijk = 0; ijk < NxNy; ijk++) {
        wdz[ijk] = exp(-w[ijk] * ds2);
    }

    if (sign == 1) {
        for (ijk = 0; ijk < NxNy; ijk++) {
            g[ijk * ns1] = qInt[ijk];
        }

        for (iz = 1; iz <= ns; iz++) {
            for (ijk = 0; ijk < NxNy; ijk++) {
                in[ijk] = g[ijk * ns1 + iz - 1] * wdz[ijk];
            }

            fftw_execute(p_forward);

            // printf("%g %g %g\n",in[0],wdz[0],g[iz-1]);

            for (i = 0; i < Nx; i++)
                for (j = 0; j < Nyh1; j++) {
                    ijk  = (long)(i * Nyh1 + j);
                    ijkr = (long)(i * Ny + j);

                    out[ijk][0] *= kxyzdz[ijkr];  // out[].re or .im for fftw2
                    out[ijk][1] *= kxyzdz[ijkr];  // out[][0] or [1] for fftw3
                }

            fftw_execute(p_backward);

            for (ijk = 0; ijk < NxNy; ijk++)
                g[ijk * ns1 + iz] = in[ijk] * wdz[ijk] / NxNy;

            // exit(0);
        }
    }
    else {
        for (ijk = 0; ijk < NxNy; ijk++) {
            g[ijk * ns1 + ns] = qInt[ijk];
        }

        for (iz = ns - 1; iz >= 0; iz--) {
            for (ijk = 0; ijk < NxNy; ijk++) {
                in[ijk] = g[ijk * ns1 + iz + 1] * wdz[ijk];
            }

            fftw_execute(p_forward);

            for (i = 0; i < Nx; i++)
                for (j = 0; j < Nyh1; j++) {
                    ijk  = (long)(i * Nyh1 + j);
                    ijkr = (long)(i * Ny + j);

                    out[ijk][0] *= kxyzdz[ijkr];
                    out[ijk][1] *= kxyzdz[ijkr];
                }

            fftw_execute(p_backward);

            for (ijk = 0; ijk < NxNy; ijk++)
                g[ijk * ns1 + iz] = in[ijk] * wdz[ijk] / NxNy;
        }
    }
}
void write_stress(double* stress_tensor)
{
    int   i, j, k;
    long  ijk;
    FILE* fp = fopen("stress_xy.txt", "w");

    for (ijk = 0; ijk < NxNy; ijk++) {
        fprintf(fp, "%lf %lf %lf %lf\n", stress_tensor[ijk], 0.0, 0.0, 0.0);
    }

    fclose(fp);
}

void init_w(double* wA, double* wB)
{
    char comment[200];
    long ijk;

    if (intag == 1028) {
        char density_name[200];
        sprintf(density_name, "phi_1.dat");

        FILE* fp = fopen(density_name, "r");

        fgets(comment, 200, fp);
        fgets(comment, 200, fp);

        for (ijk = 0; ijk < NxNy; ijk++) {
            double e1, e2, e3, wc;
            fscanf(fp, "%lf %lf %lf %lf %lf %lf\n", &e1, &e2, &e3, &wA[ijk],
                   &wB[ijk], &wc);

            wB[ijk] += wc;
        }

        fclose(fp);
    }
    else if (intag == 1024) {
        char density_name[200];
        sprintf(density_name, "phi.txt");

        FILE* fp = fopen(density_name, "r");

        for (ijk = 0; ijk < NxNy; ijk++) {
            double e1, e2;
            fscanf(fp, "%lf %lf %lf %lf\n", &e1, &e2, &wA[ijk], &wB[ijk]);
        }

        fclose(fp);
    }
}

//********************Output configuration******************************

void allocate_mem_global(double* wA, double* wB, double* phA, double* phB)
{
    time_t ts;
    int    i, j;

    int iseed = time(&ts);
    srand48(iseed);

    in  = (double*)malloc(sizeof(double) * NxNy); /* for fftw3 */
    out = (fftw_complex*)malloc(sizeof(fftw_complex) * NxNy);

    p_forward  = fftw_plan_dft_r2c_2d(Nx, Ny, in, out, FFTW_ESTIMATE);
    p_backward = fftw_plan_dft_c2r_2d(Nx, Ny, out, in, FFTW_ESTIMATE);

    FILE*  fp = fopen("para", "r");
    double fA, fB;

    fscanf(fp, "%d", &intag);  // in=1: inputing configuration is given;
    fscanf(fp, "%lf", &hAB);
    fscanf(fp, "%lf", &fA);
    fscanf(fp, "%lf, %lf", &lx, &ly);
    fscanf(fp, "%s", FEname);  // output file name for parameters;
    fscanf(fp, "%s", phname);  // output file name for configuration;
    fscanf(fp, "%lf", &ds0);
    fscanf(fp, "%d", &Narm);
    fclose(fp);

    ds0 = 0.01;
    ds2 = ds0 / 2;

    kxyzdz = (double*)malloc(sizeof(double) * NxNy);
    wdz    = (double*)malloc(sizeof(double) * NxNy);

    fB = 1.0 - fA;

    dx = lx / Nx;
    dy = ly / Ny;

    NsA = ((int)(fA / ds0 + 1.0e-8));
    NsB = ((int)(fB / ds0 + 1.0e-8));

    nsA = 1;
    nsB = 0;

    // printf("%d %d %d %d\n",NsA,NsB,nsA,nsB);
    //**************************definition of surface field and
    // confinement***********************

    for (i = 0; i <= Nx / 2 - 1; i++)
        kx[i] = 2 * Pi * i * 1.0 / Nx / dx;
    for (i = Nx / 2; i < Nx; i++)
        kx[i] = 2 * Pi * (i - Nx) * 1.0 / dx / Nx;
    for (i = 0; i < Nx; i++)
        kx[i] *= kx[i];

    for (j = 0; j <= Ny / 2 - 1; j++)
        ky[j] = 2 * Pi * j * 1.0 / Ny / dy;
    for (j = Ny / 2; j < Ny; j++)
        ky[j] = 2 * Pi * (j - Ny) * 1.0 / dy / Ny;
    for (j = 0; j < Ny; j++)
        ky[j] *= ky[j];

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++) {
            long   ijk  = (long)(i * Ny + j);
            double ksq  = kx[i] + ky[j];
            kxyzdz[ijk] = exp(-ds0 * ksq);
        }
}
