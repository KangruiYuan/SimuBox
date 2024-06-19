#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <fftw3.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

// max iter
// #define MaxIT 5000
int MaxIT;

// The program is for 2D cal, the z para is used only to draw pic
// #define Nx 128
// #define Ny 128
int    Nx;
int    Ny;
int    Nz = 1;
double dz = 1.0;
#define NxNy (Nx * Ny)

// #define Nxh1 (Nx / 2 + 1)
#define Nyh1 (Ny / 2 + 1)
#define NxNy1 (Nx * Nyh1)

#define Pi 3.1415926535897932384626433832795

#define N_hist 50
/* Parameters used in Anderson convergence */
#define Del(k, i, n) del[(i) + NxNy * (n) + N_hist * NxNy * (k)]
#define Outs(k, i, n) outs[(i) + NxNy * (n) + N_hist * NxNy * (k)]
#define U(n, m) up[(m - 1) + (N_rec - 1) * (n - 1)]
#define V(n) vp[n - 1]
#define A(n) ap[n - 1]

void initW_C4_with_array(double* wA, double* wB);
void initW_C4_with_Rg0(double* wA, double* wB);
// void initW_p6(double* wA, double* wB);
// void initW_lam(double* wA, double* wB);
void initW(double* wA, double* wB);

double min(double a, double b, double c, double d);

double freeE(double* wA, double* wB, double* phA, double* phB, double* eta);
double getConc(double* phlA, double* phlB, double phs0, double* wA, double* wB);
double
getConcBridge(double* phlA, double* phlB, double phs0, double* wA, double* wB);
void   sovDifFft(double* g,
                 double* w,
                 double* qInt,
                 double  z,
                 int     nz,
                 int     sign,
                 double  epK);
void   write_ph(double* phA, double* phB, double* wA, double* wB);
double error_cal(double* waDiffs, double* wbDiffs, double* wAs, double* wBs);
void   update_flds_hist(double* waDiff,
                        double* wbDiff,
                        double* wAnew,
                        double* wBnew,
                        double* del,
                        double* outs);
void   Anderson_mixing(double* del,
                       double* outs,
                       int     N_rec,
                       double* wA,
                       double* wB);
double cfunc_square(int x, int y);

// kryuan
// double *jp_b1, *jp_b2, *jp_ai;
// H logs the fraction of wall in a single pixel
// double* H, *old_H;
double* H;
// Posi stands for the state of the pixel whether to be poly or wall
int* Posi;
// Hw is the total fraction of wall
// Hp is the total fraction of polymer
double Hw, Hp;
// kryuan need to put into para file
double hAw, hBw;
// set the truncation fraction is 0.96 (near the wall)
double trunc_fr = 1.00;
// isolation layer is between 0.1-0.2 Rg, I set 0.15 Rg
double iso_dis = 0.15;

// the above paras are new for the wall
// double kx[Nx], ky[Ny];
double *kx, *ky;
double *kxy, dx, dy, lx, ly;
int     N_star;
int     Nlinshi;
double  dlinshi;
double  hAB, fB1, fB2, fAi, fA, fB;
char    FEname[50], phname[50];
double  epA, epB, ds0;

int ZDIMM, NsAi, NsB1, NsB2, NsA, NsB;

// added by kryuan
int    constraint;
int    bridge;
int    follow;
double Rg0;
int    arr_r, arr_c;
double wopt, wcmp;
int    in;

double  vcell, loopratio, fflloop;
double  loopratio_test, bridgeratio;
double *groupid, *qbarv;
int     in;
double  ql;

// (BBBBBBBBBBB-AAAAAAAAAAAAAA-BBBBBBBBBBBB)n

// (B1-Ai-B2)*N_star

int main(int argc, char** argv)
{
    double  R, Rs, V0, epsln, xi, yi, rij;
    double *wA, *wB, *eta, *phA, *phB;
    double  e1, Dh, e2, e3, e4, e5, e6, step = 0.050;
    double  temp;
    double  wat, wbt;
    double  LL, LH, dx0, dy0, Ddx, Ddy, lambda = 0.50;
    int     nz, i, j, k, ipcs, iseed = -3, tag;
    int     temp_arrr, temp_arrc;
    long    ijk, ijk0;
    FILE*   fp;
    time_t  ts;
    iseed = time(&ts);
    srand48(iseed);
    clock_t start, finish;

    start = clock();

    /************* x along the cylinder ***************/
    // printf("here!\n");
    fp = fopen("para", "r");
    fscanf(fp, "in=%d\n", &in);  // to how to initial the fields
    // to control constraint
    fscanf(fp, "constraint=%d\n", &constraint);
    fscanf(fp, "bridge=%d\n", &bridge);
    fscanf(fp, "follow=%d\n", &follow);
    // kryuan maybe use xAB*N replace hAB
    fscanf(fp, "hAB=%lf\n", &hAB);
    fscanf(fp, "hAw=%lf\n", &hAw);
    fscanf(fp, "hBw=%lf\n", &hBw);
    // fB1 is the free end of an arm
    // N_star is num of arms
    // f_Ai is the A fraction of arm i
    fscanf(fp, "fBend=%lf\n", &fB1);
    fscanf(fp, "Nstar=%d\n", &N_star);
    fscanf(fp, "fAi=%lf\n", &fAi);
    fscanf(fp, "Nx=%d\n", &Nx);
    fscanf(fp, "Ny=%d\n", &Ny);
    fscanf(fp, "lx=%lf\n", &lx);
    fscanf(fp, "ly=%lf\n", &ly);
    fscanf(fp, "arrr=%d\n", &arr_r);
    fscanf(fp, "arrc=%d\n", &arr_c);
    fscanf(fp, "Rg=%lf\n", &Rg0);
    fscanf(fp, "wopt=%lf\n", &wopt);
    fscanf(fp, "wcmp=%lf\n", &wcmp);
    // FEname logs the paras
    fscanf(fp, "FEname=%s\n", FEname);
    // phname logs the field
    fscanf(fp, "phname=%s\n", phname);
    fscanf(fp, "ds=%lf\n", &ds0);
    fscanf(fp, "epA=%lf\n", &epA);
    fscanf(fp, "epB=%lf\n", &epB);
    fscanf(fp, "MaxIT=%d\n", &MaxIT);
    fclose(fp);

    if (bridge) {
        MaxIT = 2;
    }

    // initialize some device space for parameters
    wA = (double*)malloc(sizeof(double) * NxNy);
    wB = (double*)malloc(sizeof(double) * NxNy);

    phA = (double*)malloc(sizeof(double) * NxNy);
    phB = (double*)malloc(sizeof(double) * NxNy);
    H   = (double*)malloc(sizeof(double) * NxNy);
    // old_H    = (double*)malloc(sizeof(double) * NxNy);
    Posi = (int*)malloc(sizeof(double) * NxNy);

    eta = (double*)malloc(sizeof(double) * NxNy);
    kx  = (double*)malloc(sizeof(double) * Nx);
    ky  = (double*)malloc(sizeof(double) * Ny);
    kxy = (double*)malloc(sizeof(double) * NxNy);

    groupid = (double*)malloc(sizeof(double) * NxNy);
    qbarv   = (double*)malloc(sizeof(double) * NxNy);

    // jp_b1 = (double*)malloc(sizeof(double) * NxNy);
    // jp_b2 = (double*)malloc(sizeof(double) * NxNy);
    // jp_ai = (double*)malloc(sizeof(double) * NxNy);

    fA  = N_star * fAi;
    fB  = 1.0 - fA;
    fB2 = 1.0 / N_star - fAi - fB1;

    NsA  = ((int)(fA / ds0 + 1.0e-6));
    NsB  = ((int)(fB / ds0 + 1.0e-6));
    NsB1 = ((int)(fB1 / ds0 + 1.0e-6));
    NsB2 = ((int)(fB2 / ds0 + 1.0e-6));
    NsAi = ((int)(fAi / ds0 + 1.0e-6));

    if (follow == 1) {
        lx = ly * Nx / Ny;
    }

    // grid spacing in the direction x; note the dz=1.0
    dx = lx / Nx;
    dy = ly / Ny;

    printf("xAB=%lf, xAw=%lf, xBw=%lf\n", hAB, hAw, hBw);
    printf("fB1=%lf, N_star=%d, fB2=%lf, fAi=%lf\n", fB1, N_star, fB2, fAi);
    printf("Nx=%d, NsB1=%d, NsB2=%d, NsAi=%d\n", Nx, NsB1, NsB2, NsAi);

    fp = fopen("para_input.txt", "w");
    // printf("here!\n");
    fprintf(fp, "Nx = %d Ny = %d\n", Nx, Ny);
    fprintf(fp, "xAB=%lf, xAw=%lf, xBw=%lf\n", hAB, hAw, hBw);
    fprintf(fp, "fB1=%lf, N_star=%d, fB2=%lf, fAi=%lf\n", fB1, N_star, fB2,
            fAi);
    fprintf(fp, "lx=%.6lf, ly=%.6lf\n", lx, ly);
    fprintf(fp, "dx=%.6lf, dy=%.6lf\n", dx, dy);
    fprintf(fp, "arr_r=%d, arr_c=%d\n", arr_r, arr_c);
    fprintf(fp, "Rg0=%lf\n", Rg0);
    fprintf(fp, "the calculated phase is %d\n", in);
    printf("the calculated phase is %d\n", in);
    if (constraint == 1) {
        fprintf(fp, "the constraint is on!\n");
        printf("the constraint is on!\n");
    }
    if (in == 10 || in == 11 || in == 12) {
        fprintf(fp, "read the fields from FILES\n");
        printf("read the fields from FILES\n");
    }
    fclose(fp);

    for (i = 0; i <= Nx / 2 - 1; i++)
        kx[i] = 2 * Pi * i * 1.0 / lx;
    for (i = Nx / 2; i < Nx; i++)
        kx[i] = 2 * Pi * (i - Nx) * 1.0 / lx;
    for (i = 0; i < Nx; i++)
        kx[i] *= kx[i];
    for (i = 0; i <= Ny / 2 - 1; i++)
        ky[i] = 2 * Pi * i * 1.0 / ly;
    for (i = Ny / 2; i < Ny; i++)
        ky[i] = 2 * Pi * (i - Ny) * 1.0 / ly;
    for (i = 0; i < Ny; i++)
        ky[i] *= ky[i];

    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++) {
            ijk      = i * Ny + j;
            kxy[ijk] = kx[i] + ky[j];
        }

    /***************Initialize Square wall field**************/
    if (constraint == 1) {
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk    = i * Ny + j;
                H[ijk] = cfunc_square(i, j);
                if (H[ijk] >= 1.0) {
                    Posi[ijk] = 0;
                }
                else {
                    Posi[ijk] = 1;
                }
                Hw += H[ijk];
            }

        Hw = Hw / NxNy;
        Hp = 1.0 - Hw;
        printf("hp = %lf\n", Hp);

        fp = fopen("./square.bin", "wb");
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk = i * Ny + j;
                // fprintf(fp, "%lf", H[ijk]);
                fwrite(&H[ijk], 8, 1, fp);
            }
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk = i * Ny + j;
                // fprintf(fp, "%lf", H[ijk] * hAw);
                temp = H[ijk] * hAw;
                fwrite(&temp, 8, 1, fp);
            }
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk = i * Ny + j;
                // fprintf(fp, "%lf", H[ijk] * hBw);
                temp = H[ijk] * hBw;
                fwrite(&temp, 8, 1, fp);
            }
        fclose(fp);
    }
    else {
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk       = i * Ny + j;
                H[ijk]    = 0;
                Posi[ijk] = 1;
            }
        iso_dis = 0;
        Hw      = 0.0;
        Hp      = 1.0;
        printf("reset iso_dis = 0\n");
        printf("hp = 1.0\n");
    }

    /***************Initialize wA, wB******************/
    if (in == 0)
        initW(wA, wB);
    else if (in == 1) {
        initW_C4_with_array(wA, wB);
    }
    else if (in == 2) {
        initW_C4_with_Rg0(wA, wB);
    }
    else if (in == 10) {
        printf("this FILE comes from old C program\n");
        fp = fopen("phin.txt", "r");
        fscanf(fp, "Nx=%d, Ny=%d\n", &Nlinshi, &Nlinshi);
        fscanf(fp, "dx=%lf, dy=%lf\n", &dlinshi, &dlinshi);
        for (j = 0; j < Ny; j++)
            for (i = 0; i < Nx; i++) {
                ijk = j * Nx + i;
                fscanf(fp, "%lf %lf %lf %lf\n", &e3, &e4, &wA[ijk], &wB[ijk]);
            }
        fclose(fp);
    }
    else if (in == 11) {
        printf("this FILE comes from CPU TOPS\n");
        fp = fopen("phin.txt", "r");
        fscanf(fp, "%d %d %d\n", &Nlinshi, &Nlinshi, &Nlinshi);
        // fscanf(fp, "dx=%lf, dy=%lf, dz=%lf\n", &dlinshi, &dlinshi, &dlinshi);
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk = j + i * Ny;
                fscanf(fp, "%lf %lf %lf %lf\n", &e3, &e4, &wA[ijk], &wB[ijk]);
            }
        fclose(fp);
    }
    else if (in == 12) {
        printf("this FILE comes from my C program\n");
        if (arr_r == -1 || arr_c == -1) {
            arr_r = (int)((lx - iso_dis * 2) / Rg0);
            arr_c = (int)((ly - iso_dis * 2) / Rg0);
        }
        fp = fopen("phin.txt", "r");
        fscanf(fp, "%d %d %d\n", &Nlinshi, &Nlinshi, &Nlinshi);
        fscanf(fp, "%lf %lf %lf\n", &dlinshi, &dlinshi, &dlinshi);
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk = i * Ny + j;
                fscanf(fp, "%lf %lf %lf %lf\n", &e3, &e4, &wA[ijk], &wB[ijk]);
            }
        fclose(fp);
    }
    else if (in == 13) {
        fp = fopen("phin_with_mask.txt", "r");
        int linshi;
        fscanf(fp, "%d %d %d\n", &linshi, &linshi, &linshi);
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++) {
                ijk = i * Ny + j;
                fscanf(fp, "%lf %lf %lf %lf %lf\n", &e1, &e1, &e2, &e3, &e5);
                wA[ijk] = e2;
                wB[ijk] = e3;
                if (e5 == 1) {
                    groupid[ijk] = 1;
                    vcell += groupid[ijk];
                }
                else
                    groupid[ijk] = 0;
            }

        fclose(fp);
        printf("vcell = %lf\n", vcell);
        printf("done\n");
    }

    // write_ph(phA,phB,wA,wB);

    fp = fopen(FEname, "w");
    fprintf(fp, "para: hAB %lf\n", hAB);
    fprintf(fp, "para: hAw %lf\n", hAw);
    fprintf(fp, "para: hBw %lf\n", hBw);
    fprintf(fp, "para: fAi %lf\n", fAi);
    fprintf(fp, "para: fB1 %lf\n", fB1);
    fprintf(fp, "para: Nstar %d\n", N_star);
    fprintf(fp, "para: lx %lf\n", lx);
    fprintf(fp, "para: ly %lf\n", ly);
    fprintf(fp, "para: arr_r %d\n", arr_r);
    fprintf(fp, "para: arr_c %d\n", arr_c);
    fprintf(fp, "para: in %d\n", in);
    fclose(fp);

    e1 = freeE(wA, wB, phA, phB, eta);
    free(wA);
    free(wB);
    free(phA);
    free(phB);
    free(eta);
    free(kxy);
    free(H);
    free(Posi);
    // free(jp_b1);
    // free(jp_ai);
    // free(jp_b2);
    free(groupid);
    free(qbarv);

    finish = clock();

    printf("Time cost: %d\n", (finish - start) / CLOCKS_PER_SEC);

    printf("***********done***********\n");
    return 1;
}

//********************Output configuration******************************
void write_ph(double* phA, double* phB, double* wA, double* wB)
{
    int   i, j, k;
    int   ijk;
    FILE *fp, *fp1;
    fp = fopen(phname, "w");
    fprintf(fp, "%d %d %d\n", 1, Nx, Ny);
    fprintf(fp, "%lf %lf %lf\n", 1.0, lx, ly);

    // for (j = 0; j < Ny; j++) {
    //     for (i = 0; i < Nx; i++) {
    //         ijk = j * Nx + i;
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            ijk = i * Ny + j;
            fprintf(fp, "%0.7lf %0.7lf %0.7lf %0.7lf\n", phA[ijk], phB[ijk],
                    wA[ijk], wB[ijk]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void initW(double* wA, double* wB)
{
    int    i, j, k;
    long   ijk;
    double wax, wbx;
    double f_a, f_b;

    // for (j = 0; j < Ny; j++) {
    //     for (i = 0; i < Nx; i++) {
    //         ijk = j * Nx + i;
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            ijk = i * Ny + j;
            if (Posi[ijk] == 0) {
                wA[ijk] = hAw * H[ijk];
                wB[ijk] = hBw * H[ijk];
            }
            else {
                f_a     = fA + 0.020 * (drand48() - 0.50);
                f_b     = 1 - H[ijk] - f_a;
                wA[ijk] = hAB * f_b + hAw * H[ijk] + 0.020 * (drand48() - 0.50);
                wB[ijk] = hAB * f_a + hBw * H[ijk] + 0.020 * (drand48() - 0.50);
            }
        }
    }
}

void initW_C4_with_array(double* wA, double* wB)
{
    int    i, j, nn;
    long   ijk;
    double r0asq, rij;
    double xi, yi, xij, yij;
    double phat, phbt;
    int    num_total;
    double x_start, y_start;
    // xc is the x coord of the center
    double* xc;
    double* yc;

    // r for x and c for y
    num_total = arr_r * arr_c;
    printf("the total num of cylinders is %d\n", num_total);
    xc = (double*)malloc(sizeof(double) * num_total);
    yc = (double*)malloc(sizeof(double) * num_total);
    if (arr_r <= 0 || arr_c <= 0) {
        printf("lx and ly is too small so we randomly init W\n");
        initW(wA, wB);
    }
    else {
        printf("the cylinder array is %d * %d\n", arr_r, arr_c);
        // the para below is used to define the max radius of cylinder
        r0asq = lx * ly * fA * Hp / Pi / num_total;
        printf("r0asq is %lf\n", r0asq);
        if (constraint == 1) {
            x_start = lx * 0.5 - (lx - 0.30) / (arr_r + 1) * 0.5 * (arr_r - 1);
            y_start = ly * 0.5 - (ly - 0.30) / (arr_c + 1) * 0.5 * (arr_c - 1);
            for (i = 0; i < arr_r; i++) {
                for (j = 0; j < arr_c; j++) {
                    ijk     = i * arr_c + j;
                    xc[ijk] = x_start + i * (lx - 0.30) / (arr_r + 1);
                    yc[ijk] = y_start + j * (ly - 0.30) / (arr_c + 1);
                    // printf("%lf, %lf\n", xc[ijk], yc[ijk]);
                }
            }
        }
        else {
            x_start = 0.0;
            y_start = 0.0;
            for (i = 0; i < arr_r; i++) {
                for (j = 0; j < arr_c; j++) {
                    ijk     = i * arr_c + j;
                    xc[ijk] = x_start + i * lx / (arr_r - 1);
                    yc[ijk] = y_start + j * ly / (arr_c - 1);
                    // printf("%lf, %lf\n", xc[ijk], yc[ijk]);
                }
            }
        }

        for (i = 0; i < Nx; i++) {
            xi = i * dx;
            for (j = 0; j < Ny; j++) {
                yi  = j * dy;
                ijk = i * Ny + j;

                if (Posi[ijk] == 0) {
                    wA[ijk] = hAw * H[ijk];
                    wB[ijk] = hBw * H[ijk];
                }
                else {
                    phat = 0.0;
                    phbt = 1.0;

                    for (nn = 0; nn < num_total; nn++) {
                        xij = xi - xc[nn];
                        yij = yi - yc[nn];

                        rij = xij * xij + yij * yij;
                        // printf("%4d %4d rij is %lf\n", i, j, rij);

                        if (rij <= r0asq) {
                            // printf("rij is %lf\n", rij);
                            phat = 1.0;
                            phbt = 0.0;
                        }
                    }
                    wA[ijk] = hAB * phbt * (1 - H[ijk]) + hAw * H[ijk] +
                              0.020 * (drand48() - 0.50);
                    wB[ijk] = hAB * phat * (1 - H[ijk]) + hBw * H[ijk] +
                              0.020 * (drand48() - 0.50);
                }
            }
        }
    }
    free(xc);
    free(yc);
}

void initW_C4_with_Rg0(double* wA, double* wB)
{
    int  i, j, nn;
    long ijk;
    // xc is the x coord of the center
    // double xc[4], yc[4], r0asq, rij;
    double r0asq, rij;
    double xi, yi, xij, yij;
    double phat, phbt;
    // int    num_r, num_c, num_total;
    int    num_total;
    double x_start, y_start;
    // round with some problem
    double* xc;
    double* yc;

    arr_r     = (int)((lx - iso_dis * 2) / Rg0);
    arr_c     = (int)((ly - iso_dis * 2) / Rg0);
    num_total = arr_r * arr_c;
    xc        = (double*)malloc(sizeof(double) * num_total);
    yc        = (double*)malloc(sizeof(double) * num_total);
    if (arr_c <= 0 || arr_r <= 0) {
        printf("lx and ly is too small so we randomly init W\n");
        initW(wA, wB);
    }
    else {
        printf("the cylinder array is %d * %d\n", arr_r, arr_c);
        x_start = lx * 0.5 - Rg0 * 0.5 * (arr_r - 1);
        y_start = ly * 0.5 - Rg0 * 0.5 * (arr_c - 1);
        // the para below is used to define the max radius of cylinder
        r0asq = lx * ly * fA * Hp / Pi / num_total;
        printf("r0asq is %lf\n", r0asq);

        for (i = 0; i < arr_r; i++) {
            for (j = 0; j < arr_c; j++) {
                ijk     = i * arr_c + j;
                xc[ijk] = x_start + i * Rg0;
                yc[ijk] = y_start + j * Rg0;
            }
        }

        for (i = 0; i < Nx; i++) {
            xi = i * dx;
            for (j = 0; j < Ny; j++) {
                yi  = j * dy;
                ijk = i * Ny + j;

                if (Posi[ijk] == 0) {
                    wA[ijk] = hAw * H[ijk];
                    wB[ijk] = hBw * H[ijk];
                }
                else {
                    phat = 0.0;
                    phbt = 1.0;

                    for (nn = 0; nn < num_total; nn++) {
                        xij = xi - xc[nn];
                        yij = yi - yc[nn];

                        rij = xij * xij + yij * yij;
                        // printf("rij is %lf\n", rij);

                        if (rij <= r0asq) {
                            // printf("rij is %lf\n", rij);

                            phat = 1.0;
                            phbt = 0.0;
                        }
                    }
                    wA[ijk] = hAB * phbt * (1 - H[ijk]) + hAw * H[ijk] +
                              0.020 * (drand48() - 0.50);
                    wB[ijk] = hAB * phat * (1 - H[ijk]) + hBw * H[ijk] +
                              0.020 * (drand48() - 0.50);
                }
            }
        }
    }
    free(xc);
    free(yc);
}

double freeE(double* wA, double* wB, double* phA, double* phB, double* eta)
{
    int    i, j, ii, k, iter, tag;
    long   ijk;
    double freeEnergy, freeOld, phbA, phbB, qC;
    double freeW, freeAB, freeS, freeDiff, *feW, *feAB, *feWS;
    // new
    double freeAW, freeBW, freeWsurf;
    double Sm1, Sm2, psum, *psuC;
    // double  beta;
    double *waDiff, *wbDiff, InCompMax;
    double *del, *outs, *wAnew, *wBnew, err;
    int     N_rec;
    FILE*   fp;
    // int     flag;
    FILE *fp_b1, *fp_b2, *fp_ai;

    psuC   = (double*)malloc(sizeof(double) * NxNy);
    waDiff = (double*)malloc(sizeof(double) * NxNy);
    wbDiff = (double*)malloc(sizeof(double) * NxNy);

    wAnew = (double*)malloc(sizeof(double) * NxNy);
    wBnew = (double*)malloc(sizeof(double) * NxNy);

    del  = (double*)malloc(sizeof(double) * N_hist * 2 * NxNy);
    outs = (double*)malloc(sizeof(double) * N_hist * 2 * NxNy);

    for (ijk = 0; ijk < N_hist * 2 * NxNy; ijk++) {
        del[ijk]  = 0.0;
        outs[ijk] = 0.0;
    }

    // sm1 for InCompMax
    Sm1 = 1.0e-8;
    // sm2 for freeEdiff
    Sm2 = 1.0e-10;
    // maxIter = MaxIT;
    // wopt    = 0.065;
    // wcmp    = 0.06;
    // beta    = 1.0;

    iter = 0;

    freeEnergy = 0.0;

    do {
        iter = iter + 1;
        // printf("%d\n", iter);

        // kryuan
        if (iter == 1) {
            qC = getConc(phA, phB, ds0, wA, wB);
        }
        else {
            if (bridge == 0) {
                qC = getConc(phA, phB, ds0, wA, wB);
            }
            else {
                qC = getConcBridge(phA, phB, ds0, wA, wB);
            }
        }

        if (bridge) {
            double qsum = 0;
            for (ijk = 0; ijk < NxNy; ijk++) {
                qsum += qbarv[ijk];
                qbarv[ijk] = qbarv[ijk] * groupid[ijk];
                // qbarv2[ijk] = qbarv2[ijk] * groupid[ijk];
            }
            printf("qsum=%lf\n", qsum);
        }

        freeW     = 0.0;
        freeAB    = 0.0;
        freeS     = 0.0;
        freeWsurf = 0.0;
        InCompMax = 0.0;

        for (i = 0; i < Nx; i++) {
            for (j = 0; j < Ny; j++) {
                ijk = i * Ny + j;

                eta[ijk] =
                    (wA[ijk] + wB[ijk] - hAB - H[ijk] * (hAw + hBw - hAB)) /
                    2.0;

                if (Posi[ijk] == 0) {
                    psum = 0.0;
                }
                else {
                    // if (iter < 3)
                    //     printf("%lf, %lf, %lf\n", phA[ijk], phB[ijk],
                    //     H[ijk]);
                    psum = 1.0 - phA[ijk] - phB[ijk] - H[ijk];
                }

                // psum = 1.0 - phA[ijk] - phB[ijk];
                psuC[ijk] = psum;

                if (fabs(psum) > InCompMax)
                    InCompMax = fabs(psum);
                wAnew[ijk] = hAB * phB[ijk] + hAw * H[ijk] + eta[ijk];
                wBnew[ijk] = hAB * phA[ijk] + hBw * H[ijk] + eta[ijk];

                waDiff[ijk] = wAnew[ijk] - wA[ijk];
                wbDiff[ijk] = wBnew[ijk] - wB[ijk];
                waDiff[ijk] -= wcmp * psum;
                wbDiff[ijk] -= wcmp * psum;

                freeWsurf = freeWsurf + hAw * phA[ijk] * H[ijk] +
                            hBw * phB[ijk] * H[ijk];

                freeAB = freeAB + hAB * phA[ijk] * phB[ijk];
                freeW  = freeW - wA[ijk] * phA[ijk] - wB[ijk] * phB[ijk] -
                        eta[ijk] * psum;
            }
        }

        freeAB /= Hp * NxNy;
        freeW /= Hp * NxNy;
        freeWsurf /= Hp * NxNy;
        freeS = -log(qC);

        freeOld    = freeEnergy;
        freeEnergy = freeAB + freeW + freeS + freeWsurf;

        // judge the error
        err = error_cal(waDiff, wbDiff, wA, wB);

        // update the history fields, and zero is new fields
        update_flds_hist(waDiff, wbDiff, wAnew, wBnew, del, outs);

        if (err > 0.01 || iter < 300) {
            // flag = 0;
            for (ijk = 0; ijk < NxNy; ijk++) {
                wA[ijk] += wopt * waDiff[ijk];
                wB[ijk] += wopt * wbDiff[ijk];
            }
        }
        else {
            N_rec = (iter - 1) < N_hist ? (iter - 1) : N_hist;
            Anderson_mixing(del, outs, N_rec, wA, wB);
            // flag = 1;
        }

        //**** print out the free energy and error results ****

        if (iter == 1 || iter % 10 == 0 || iter >= MaxIT) {
            if (iter == 10) {
                FILE* fp = fopen("printout_c.txt", "w");
                fclose(fp);
            }
            fp = fopen("printout_c.txt", "a");
            fprintf(fp, "%d\n", iter);
            fprintf(fp, "%10.8e, %10.8e, %10.8e, %10.8e, %10.8e, %10.8e\n",
                    freeEnergy, freeAB, freeW, freeS, freeWsurf, InCompMax);
            printf("%5d: %10.8e, %10.8e, %10.8e, %10.8e, %10.8e, %10.8e\n",
                   iter, freeEnergy, freeAB, freeW, freeS, freeWsurf,
                   InCompMax);
            fclose(fp);
        }

        freeDiff = fabs(freeEnergy - freeOld);
        if (iter == 1 || iter % 50 == 0)
            write_ph(phA, phB, wA, wB);
    } while (iter < MaxIT && (freeDiff > Sm2 || InCompMax > Sm1));

    fp = fopen("log_bridge.txt", "w");
    fprintf(fp, "%lf\n", bridgeratio);
    fclose(fp);

    fp = fopen("printout_c.txt", "a");
    fprintf(fp, "%d\n", iter);
    fprintf(fp, "%10.8e, %10.8e, %10.8e, %10.8e, %10.8e, %10.8e\n", freeEnergy,
            freeAB, freeW, freeS, freeWsurf, InCompMax);
    fclose(fp);

    fp = fopen(FEname, "w");
    fprintf(fp, "para: hAB %lf\n", hAB);
    fprintf(fp, "para: hAw %lf\n", hAw);
    fprintf(fp, "para: hBw %lf\n", hBw);
    fprintf(fp, "para: fAi %lf\n", fAi);
    fprintf(fp, "para: fB1 %lf\n", fB1);
    fprintf(fp, "para: Nstar %d\n", N_star);
    fprintf(fp, "para: lx %lf\n", lx);
    fprintf(fp, "para: ly %lf\n", ly);
    fprintf(fp, "para: in %d\n", in);
    fprintf(fp, "result: arr_r %d\n", arr_r);
    fprintf(fp, "result: arr_c %d\n", arr_c);
    fprintf(fp, "result: freeEnergy %.8e\n", freeEnergy);
    fprintf(fp, "result: freeAB %.8e\n", freeAB);
    fprintf(fp, "result: freeW %.8e\n", freeW);
    fprintf(fp, "result: freeS %.8e\n", freeS);
    fprintf(fp, "result: freeWsurf %.8e\n", freeWsurf);
    fprintf(fp, "result: InCompMax %.8e\n", InCompMax);
    fprintf(fp, "result: freeDiff %.8e\n", freeDiff);
    fclose(fp);

    write_ph(phA, phB, wA, wB);

    // fp_b1 = fopen("./joint_b1.bin", "wb");
    // fp_b2 = fopen("./joint_b2.bin", "wb");
    // fp_ai = fopen("./joint_ai.bin", "wb");
    // for (ijk = 0; ijk < NxNy; ijk++) {
    //     fwrite(&jp_b1[ijk], 8, 1, fp_b1);
    //     fwrite(&jp_b2[ijk], 8, 1, fp_b2);
    //     fwrite(&jp_ai[ijk], 8, 1, fp_ai);
    // }
    // fclose(fp_b1);
    // fclose(fp_b2);
    // fclose(fp_ai);

    free(psuC);
    free(waDiff);
    free(wbDiff);
    free(wAnew);
    free(wBnew);
    free(del);
    free(outs);
    return freeDiff;
}

double getConc(double* phlA, double* phlB, double phs0, double* wA, double* wB)
{
    int     i, j, k, iz, tag, nn;
    long    ijk, ijkiz;
    double *qB1, *qcB1, *qB2, *qcB2, *qAi, *qcAi;
    double  ql, ffl, fflA, fflB, *qInt, qtmp;

    double fflAi, fflB1, fflB2;
    double dzAi, dzB1, dzB2;

    //(BBBBBB-AAAAAAA-BBBBBB)n    (B1-Ai-B2)n  Star
    qB1  = (double*)malloc(sizeof(double) * NxNy * (NsB1 + 1));
    qcB1 = (double*)malloc(sizeof(double) * NxNy * (NsB1 + 1));
    qB2  = (double*)malloc(sizeof(double) * NxNy * (NsB2 + 1));
    qcB2 = (double*)malloc(sizeof(double) * NxNy * (NsB2 + 1));
    qAi  = (double*)malloc(sizeof(double) * NxNy * (NsAi + 1));
    qcAi = (double*)malloc(sizeof(double) * NxNy * (NsAi + 1));

    qInt = (double*)malloc(sizeof(double) * NxNy);
    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = 1.0;
    }
    sovDifFft(qB1, wB, qInt, fB1, NsB1, 1, epB);  // 0 to fB1 for qB1

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = qB1[ijk * (NsB1 + 1) + NsB1];
    }
    sovDifFft(qAi, wA, qInt, fAi, NsAi, 1, epA);  // for qAi

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = qAi[ijk * (NsAi + 1) + NsAi];
    }
    if (bridge) {
        for (ijk = 0; ijk < NxNy; ijk++) {
            qbarv[ijk] = qAi[ijk * (NsAi + 1) + NsAi];
        }
    }

    sovDifFft(qB2, wB, qInt, fB2, NsB2, 1, epB);  // for qB2

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = pow(qB2[ijk * (NsB2 + 1) + NsB2], (N_star - 1));
    }
    sovDifFft(qcB2, wB, qInt, fB2, NsB2, -1, epB);  // for qcB2

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = qcB2[ijk * (NsB2 + 1)];
    }

    sovDifFft(qcAi, wA, qInt, fAi, NsAi, -1, epA);  // for qcAi
    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = qcAi[ijk * (NsAi + 1)];
    }

    sovDifFft(qcB1, wB, qInt, fB1, NsB1, -1, epB);  // for qcB1
    ql = 0.0;
    for (ijk = 0; ijk < NxNy; ijk++) {
        ql += qcB1[ijk * (NsB1 + 1)];
    }

    ql /= NxNy * Hp;
    // phs0 = ds0;
    ffl  = 1.0 / ql;
    dzAi = fAi / NsAi;
    dzB1 = fB1 / NsB1;
    dzB2 = fB2 / NsB2;

    fflAi = dzAi * ffl;
    fflB1 = dzB1 * ffl;
    fflB2 = dzB2 * ffl;

    for (ijk = 0; ijk < NxNy; ijk++) {
        phlA[ijk] = 0.0;
        phlB[ijk] = 0.0;

        ZDIMM = NsAi + 1;
        for (iz = 0; iz <= NsAi; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsAi)
                phlA[ijk] += (0.50 * qAi[ijkiz] * qcAi[ijkiz]);
            else
                phlA[ijk] += (qAi[ijkiz] * qcAi[ijkiz]);
        }

        phlA[ijk] = N_star * phlA[ijk] * fflAi;
        // jp_ai[ijk] = qAi[ijk * (NsAi + 1) + NsAi] *
        //              qcAi[ijk * (NsAi + 1) + NsAi] * N_star * ffl;

        ZDIMM = NsB1 + 1;
        qtmp  = 0.0;
        for (iz = 0; iz <= NsB1; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsB1) {
                qtmp += (0.50 * qB1[ijkiz] * qcB1[ijkiz]);
            }
            else {
                qtmp += (qB1[ijkiz] * qcB1[ijkiz]);
            }
        }

        phlB[ijk] += qtmp * N_star * fflB1;
        // phlB[ijk] = N_star * phlB[ijk];
        // jp_b1[ijk] = qB1[ijk * (NsB1 + 1) + NsB1] *
        //              qcB1[ijk * (NsB1 + 1) + NsB1] * N_star * ffl;

        ZDIMM = NsB2 + 1;
        qtmp  = 0.0;
        for (iz = 0; iz <= NsB2; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsB2)
                qtmp += (0.50 * qB2[ijkiz] * qcB2[ijkiz]);
            else
                qtmp += (qB2[ijkiz] * qcB2[ijkiz]);
        }

        phlB[ijk] += N_star * qtmp * fflB2;
        // jp_b2[ijk] =
        //     qB2[ijk * (NsB2 + 1) + NsB2] * qcB2[ijk * (NsB2 + 1) + NsB2] *
        //     ffl;

        // phlA[ijk] *= ffl;
        // phlB[ijk] *= ffl;
    }

    free(qB1);
    free(qcB1);

    free(qB2);
    free(qcB2);

    free(qAi);
    free(qcAi);
    free(qInt);

    return ql;
}

double
getConcBridge(double* phlA, double* phlB, double phs0, double* wA, double* wB)
{
    int     i, j, k, iz, tag, nn;
    long    ijk, ijkiz;
    double *qB1, *qcB1, *qB2, *qcB2, *qAi, *qcAi;
    double  ffl, fflA, fflB, *qInt, qtmp;

    double ql_1st, ql_all;

    double fflAi, fflB1, fflB2;
    double dzAi, dzB1, dzB2;

    double *qIntAi_B2, *qInt_other, *qInt_1st;

    //(BBBBBB-AAAAAAA-BBBBBB)n    (B1-Ai-B2)n  Star
    qB1  = (double*)malloc(sizeof(double) * NxNy * (NsB1 + 1));
    qcB1 = (double*)malloc(sizeof(double) * NxNy * (NsB1 + 1));
    qB2  = (double*)malloc(sizeof(double) * NxNy * (NsB2 + 1));
    qcB2 = (double*)malloc(sizeof(double) * NxNy * (NsB2 + 1));
    qAi  = (double*)malloc(sizeof(double) * NxNy * (NsAi + 1));
    qcAi = (double*)malloc(sizeof(double) * NxNy * (NsAi + 1));

    qInt       = (double*)malloc(sizeof(double) * NxNy);
    qInt_1st   = (double*)malloc(sizeof(double) * NxNy);
    qInt_other = (double*)malloc(sizeof(double) * NxNy);
    qIntAi_B2  = (double*)malloc(sizeof(double) * NxNy);

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk]      = 1.0;
        qIntAi_B2[ijk] = qbarv[ijk];
    }

    // for normal arm
    sovDifFft(qB1, wB, qInt, fB1, NsB1, 1, epB);  // 0 to fB1 for qB1

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = qB1[ijk * (NsB1 + 1) + NsB1];
    }
    sovDifFft(qAi, wA, qInt, fAi, NsAi, 1, epA);  // for qAi

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = qAi[ijk * (NsAi + 1) + NsAi];
    }

    sovDifFft(qB2, wB, qInt, fB2, NsB2, 1, epB);  // for qB2

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt_other[ijk] = pow(qB2[ijk * (NsB2 + 1) + NsB2], (N_star - 1));
    }

    // for arm in 1st cell
    sovDifFft(qB2, wB, qIntAi_B2, fB2, NsB2, 1, epB);

    for (ijk = 0; ijk < NxNy; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt_1st[ijk] = qB2[ijk * (NsB2 + 1) + NsB2];
    }

    // sovDifFft(qcB2, wB, qInt, fB2, NsB2, -1, epB);  // for qcB2

    // for (ijk = 0; ijk < NxNy; ijk++) {
    //     // ijk=(i*Ny+j)*Nz+k;
    //     qInt[ijk] = qcB2[ijk * (NsB2 + 1)];
    // }

    // sovDifFft(qcAi, wA, qInt, fAi, NsAi, -1, epA);  // for qcAi
    // for (ijk = 0; ijk < NxNy; ijk++) {
    //     // ijk=(i*Ny+j)*Nz+k;
    //     qInt[ijk] = qcAi[ijk * (NsAi + 1)];
    // }
    // sovDifFft(qcB1, wB, qInt, fB1, NsB1, -1, epB);  // for qcB1
    // ql = 0.0;
    // for (ijk = 0; ijk < NxNy; ijk++) {
    //     ql += qcB1[ijk * (NsB1 + 1)];
    // }

    // ql /= NxNy * Hp;

    ql_1st = 0.0;
    ql_all = 0.0;
    for (ijk = 0; ijk < NxNy; ijk++) {
        ql_1st += N_star * pow(qInt_1st[ijk], N_star) * groupid[ijk];
        ql_all += N_star * qInt_other[ijk] * qInt_1st[ijk] * groupid[ijk];
    }
    loopratio   = ql_1st / ql_all;
    bridgeratio = 1 - loopratio;
    printf("loopratio=%lf\nbridgeratio=%lf\n", loopratio, bridgeratio);

    // phs0 = ds0;
    // ffl = 1.0 / ql;

    // dzAi = fAi / NsAi;
    // dzB1 = fB1 / NsB1;
    // dzB2 = fB2 / NsB2;

    // fflAi = dzAi * ffl;
    // fflB1 = dzB1 * ffl;
    // fflB2 = dzB2 * ffl;

    // FILE* fp;
    // fp = fopen("Joint_B2A2.txt", "w");

    // fprintf(fp, "%d %d %d\n", 1, Nx, Ny);
    // ZDIMM = NsA2 + 1;
    // for (ijk = 0; ijk < NxNy; ijk++) {

    //     fprintf(fp, "%f\n", Joint_B2A2[ijk]);
    // }
    // fclose(fp);

    free(qB1);
    free(qcB1);

    free(qB2);
    free(qcB2);

    free(qAi);
    free(qcAi);
    free(qInt);

    free(qInt_1st);
    free(qInt_other);
    free(qIntAi_B2);

    // return ql;
    return 1.0;
}

void sovDifFft(double* g,
               double* w,
               double* qInt,
               double  z,
               int     nz,
               int     sign,
               double  epK)
{
    int           i, j, k, iz;
    unsigned long ijk, ijkr;
    double        dzc, *wdz;
    double *      kxyzdz, dzc2;
    double*       in;
    unsigned int  nnum[3];
    nnum[1] = Nx;
    nnum[2] = Ny;
    fftw_complex* out;
    fftw_plan     p_forward, p_backward;

    ZDIMM = nz + 1;

    wdz    = (double*)malloc(sizeof(double) * NxNy);
    kxyzdz = (double*)malloc(sizeof(double) * NxNy);
    in     = (double*)malloc(sizeof(double) * NxNy);

    out  = (fftw_complex*)malloc(sizeof(fftw_complex) * NxNy1);
    dzc  = z / nz;
    dzc2 = 0.50 * dzc;

    // for (j = 0; j < Ny; j++)
    //     for (i = 0; i < Nx; i++) {
    //         ijk         = j * Nx + i;
    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++) {
            ijk         = i * Ny + j;
            kxyzdz[ijk] = exp(-dzc * kxy[ijk] * epK);
            // wdz[ijk]    = 0.0;
            wdz[ijk] = exp(-w[ijk] * dzc2) * Posi[ijk];
        }

    p_forward  = fftw_plan_dft_r2c_2d(Nx, Ny, in, out, FFTW_ESTIMATE);
    p_backward = fftw_plan_dft_c2r_2d(Nx, Ny, out, in, FFTW_ESTIMATE);
    // p_forward  = fftw_plan_dft_r2c_2d(Ny, Nx, in, out, FFTW_ESTIMATE);
    // p_backward = fftw_plan_dft_c2r_2d(Ny, Nx, out, in, FFTW_ESTIMATE);
    if (sign == 1) {
        for (ijk = 0; ijk < NxNy; ijk++) {
            g[ijk * ZDIMM] = qInt[ijk];
        }

        for (iz = 1; iz <= nz; iz++) {
            for (ijk = 0; ijk < NxNy; ijk++) {
                in[ijk] = g[ijk * ZDIMM + iz - 1] * wdz[ijk];
            }

            fftw_execute(p_forward);

            for (i = 0; i < Nx; i++)
                for (j = 0; j < Nyh1; j++) {
                    ijk  = i * Nyh1 + j;
                    ijkr = i * Ny + j;
                    out[ijk][0] *= kxyzdz[ijkr];  // out[].re or .im for fftw2
                    out[ijk][1] *= kxyzdz[ijkr];  // out[][0] or [1] for fftw3
                }

            fftw_execute(p_backward);

            for (ijk = 0; ijk < NxNy; ijk++) {
                g[ijk * ZDIMM + iz] = in[ijk] * wdz[ijk] / NxNy;
            }
        }
    }
    else {
        for (ijk = 0; ijk < NxNy; ijk++) {
            g[ijk * ZDIMM + nz] = qInt[ijk];
        }

        for (iz = nz - 1; iz >= 0; iz--) {
            for (ijk = 0; ijk < NxNy; ijk++) {
                in[ijk] = g[ijk * ZDIMM + iz + 1] * wdz[ijk];
            }

            fftw_execute(p_forward);

            for (i = 0; i < Nx; i++)
                for (j = 0; j < Nyh1; j++) {
                    ijk  = i * Nyh1 + j;
                    ijkr = i * Ny + j;
                    out[ijk][0] *= kxyzdz[ijkr];
                    out[ijk][1] *= kxyzdz[ijkr];
                }

            fftw_execute(p_backward);

            for (ijk = 0; ijk < NxNy; ijk++) {
                g[ijk * ZDIMM + iz] = in[ijk] * wdz[ijk] / NxNy;
            }
        }
    }

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);
    free(wdz);
    free(kxyzdz);
    free(in);
    free(out);
}

double error_cal(double* waDiffs, double* wbDiffs, double* wAs, double* wBs)
{
    double err_dif, err_w, err;
    int    ijk;

    err     = 0.0;
    err_dif = 0.0;
    err_w   = 0.0;
    for (ijk = 0; ijk < NxNy; ijk++) {
        err_dif += pow(waDiffs[ijk], 2) + pow(wbDiffs[ijk], 2);
        err_w += pow(wAs[ijk], 2) + pow(wBs[ijk], 2);
    }
    err = err_dif / err_w;
    err = sqrt(err);

    return err;
}

void update_flds_hist(double* waDiff,
                      double* wbDiff,
                      double* wAnew,
                      double* wBnew,
                      double* del,
                      double* outs)
{
    int ijk, j;

    for (j = N_hist - 1; j > 0; j--) {
        for (ijk = 0; ijk < NxNy; ijk++) {
            Del(0, ijk, j) = Del(0, ijk, j - 1);
            Del(1, ijk, j) = Del(1, ijk, j - 1);

            Outs(0, ijk, j) = Outs(0, ijk, j - 1);
            Outs(1, ijk, j) = Outs(1, ijk, j - 1);
        }
        //		printf("outs[%d] = %lf\n", j, Outs(1, 10, j));
    }

    for (ijk = 0; ijk < NxNy; ijk++) {
        Del(0, ijk, 0) = waDiff[ijk];
        Del(1, ijk, 0) = wbDiff[ijk];

        Outs(0, ijk, 0) = wAnew[ijk];
        Outs(1, ijk, 0) = wBnew[ijk];
    }
    // printf("outs[0] = %lf\n", Outs(1,10,j));
    // getchar();
}

/*********************************************************************/
/*
  Anderson mixing [O(Nx)]

  CHECKED
*/

void Anderson_mixing(double* del,
                     double* outs,
                     int     N_rec,
                     double* wA,
                     double* wB)
{
    int     i, k, ijk;
    int     n, m;
    double *up, *vp, *ap;
    int     s;

    gsl_matrix_view  uGnu;
    gsl_vector_view  vGnu, aGnu;
    gsl_permutation* p;

    up = (double*)malloc(sizeof(double) * (N_rec - 1) * (N_rec - 1));
    vp = (double*)malloc(sizeof(double) * (N_rec - 1));
    ap = (double*)malloc(sizeof(double) * (N_rec - 1));

    /*
        Calculate the U-matrix and the V-vector
        Follow Shuang, and add the A and B components together.
    */

    for (n = 1; n < N_rec; n++) {
        V(n) = 0.0;
        // vp[n - 1] = 0.0;

        for (ijk = 0; ijk < NxNy; ijk++) {
            V(n) += (Del(0, ijk, 0) - Del(0, ijk, n)) * Del(0, ijk, 0);
            V(n) += (Del(1, ijk, 0) - Del(1, ijk, n)) * Del(1, ijk, 0);
            // vp[n - 1] += (Del(0, ijk, 0) - Del(0, ijk, n)) * Del(0, ijk, 0);
            // vp[n - 1] += (Del(1, ijk, 0) - Del(1, ijk, n)) * Del(1, ijk, 0);
        }

        for (m = n; m < N_rec; m++) {
            U(n, m) = 0.0;
            for (ijk = 0; ijk < NxNy; ijk++) {
                U(n, m) += (Del(0, ijk, 0) - Del(0, ijk, n)) *
                           (Del(0, ijk, 0) - Del(0, ijk, m));
                U(n, m) += (Del(1, ijk, 0) - Del(1, ijk, n)) *
                           (Del(1, ijk, 0) - Del(1, ijk, m));
            }
            U(m, n) = U(n, m);
        }
    }

    /* Calculate A - uses GNU LU decomposition for U A = V */

    uGnu = gsl_matrix_view_array(up, N_rec - 1, N_rec - 1);
    vGnu = gsl_vector_view_array(vp, N_rec - 1);
    aGnu = gsl_vector_view_array(ap, N_rec - 1);

    p = gsl_permutation_alloc(N_rec - 1);

    gsl_linalg_LU_decomp(&uGnu.matrix, p, &s);

    gsl_linalg_LU_solve(&uGnu.matrix, p, &vGnu.vector, &aGnu.vector);

    gsl_permutation_free(p);

    /* Update omega */

    for (ijk = 0; ijk < NxNy; ijk++) {
        wA[ijk] = Outs(0, ijk, 0);
        wB[ijk] = Outs(1, ijk, 0);

        for (n = 1; n < N_rec; n++) {
            wA[ijk] += A(n) * (Outs(0, ijk, n) - Outs(0, ijk, 0));
            wB[ijk] += A(n) * (Outs(1, ijk, n) - Outs(1, ijk, 0));
            // wA[ijk] += ap[n - 1] * (Outs(0, ijk, n) - Outs(0, ijk, 0));
            // wB[ijk] += ap[n - 1] * (Outs(1, ijk, n) - Outs(1, ijk, 0));
        }
    }

    free(ap);
    free(vp);
    free(up);
}

double min(double a, double b, double c, double d)
{
    double temp;
    temp = a;
    if (b < temp) {
        temp = b;
    }
    if (c < temp) {
        temp = c;
    }
    if (d < temp) {
        temp = d;
    }
    return temp;
}

double cfunc_square(int x, int y)
{
    // printf("run begin!\n");
    double temp_up, temp_left, temp_down, temp_right;
    double min_dis;
    // transform the intdex of x and y into the unit of Rg
    double px, py;
    x += 1;
    y += 1;
    px = x * dx;
    py = y * dy;

    // sigma is the dis where the wall fraction decrease to 0.5
    double sig  = 0.25;
    double lamb = 0.1;
    // double mycut = 0.15801580158;
    double wall_frac;

    if ((px <= iso_dis) || ((lx - px) <= iso_dis)) {
        return 1.0;
    }
    else if ((py <= iso_dis) || ((ly - py) <= iso_dis)) {
        return 1.0;
    }
    else {
        temp_up    = py - iso_dis;
        temp_left  = px - iso_dis;
        temp_down  = ly - py - iso_dis;
        temp_right = lx - px - iso_dis;
        min_dis    = min(temp_up, temp_left, temp_down, temp_right);
        wall_frac  = 0.5 * (1.0 - tanh((min_dis - sig) / lamb));

        // if ((wall_frac >= trunc_fr) || (min_dis < mycut)) {
        // if (wall_frac >= trunc_fr) {
        //     if (wall_frac - trunc_fr < 0.01){
        //         return wall_frac;
        //     }
        //     else if (wall_frac - 0.97 < 0.002) {
        //         return 0.7;
        //     }
        //     else {
        //         return 1.0;
        //     }
        // }
        if (wall_frac >= trunc_fr) {
            return 1.0;
        }
        else if (min_dis > (2 * sig)) {
            return 0.0;
        }
        else {
            return wall_frac;
        }
    }
}
