#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <fftw3.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#define MaxIT 2  // Maximum iteration steps
// #define Nx 192
// #define Ny 192
// #define Nz 1  // grid size
int Nx, Ny, Nz = 1;
#define NyNz (Ny * Nz)
#define NxNyNz (Nx * Ny * Nz)

#define Nzh1 (Nz / 2 + 1)
#define Nz2 (2 * Nzh1)
#define NxNyNz1 (Nx * Ny * Nzh1)

#define Pi 3.141592653589
#define N_hist 50
#define MaxNlp 20
/* Parameters used in Anderson convergence */
#define Del(k, i, n) del[(i) + NxNyNz * (n) + N_hist * NxNyNz * (k)]
#define Outs(k, i, n) outs[(i) + NxNyNz * (n) + N_hist * NxNyNz * (k)]
#define U(n, m) up[(m - 1) + (N_rec - 1) * (n - 1)]
#define V(n) vp[n - 1]
#define A(n) ap[n - 1]

void initW(double* wA, double* wB);
void initW_C(double* wA, double* wB);
void initW_L(double* wA, double* wB);
void initW_G(double* wA, double* wB);

double freeE(double* wA, double* wB, double* phA, double* phB, double* eta);
double getConc(double* phlA, double* phlB, double phs0, double* wA, double* wB);
double
getConcBridge(double* phlA, double* phlB, double phs0, double* wA, double* wB);
void sovDifFft(double* g, double* w, double* qInt, double z, int ns, int sign);
void write_ph(double* phA, double* phB, double* wA, double* wB);
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

int     ZDIMM, NsA, NsB, NsA1, NsA2, NsB1, NsB2, NsB3;
double *kx, *ky, *kz, *kxyz, dx, dy, dz;
double  lx, ly, lz, ds0, freeEnergy;
double  hAB, f, fB, ah, bh, ch, abch;  // HNA: surface field;  hAB=xAB*N;
double  tauB2, Etta, ksi;
double  fA1, fA2, fB1, fB2, fB3;
double* Joint_B2A2;
// double epA, epB, epC;

char FEname[30], phname[30];

// ****************************for Bridge*************************

double  vcell, loopratio, fflloop;
double  loopratio_test, bridgeratio;
double *groupid, *qbarv;
int     in;
double  ql;

int main(int argc, char** argv)
{
    double *wA, *wB, *eta, *phA, *phB;
    double  e1, e2, e3, e4, e5, en1, en2, en3, delt_x, lx1, lx2;
    //	double e1,e2,e3,e4,e5,e6,en1,en2, en3,delt_x,lx1,lx2;
    double rjk, yj, zk;
    int    i, i1, j, k, Nlp, iseed = -3;  // local_x_starti;
    long   ijk, ijk0;
    // MPI_Status status;
    FILE * fp, *fp1;
    time_t ts;
    iseed = time(&ts);

    srand48(iseed);

    fp = fopen("para", "r");
    fscanf(fp, "in=%d\n", &in);  // in=1: inputing configuration is given;
    fscanf(fp, "hAB=%lf\n", &hAB);
    fscanf(fp, "tau=%lf\n", &tauB2);
    fscanf(fp, "ksi=%lf\n", &ksi);
    fscanf(fp, "fA=%lf\n", &f);
    fscanf(fp, "lx=%lf\n", &lx);
    fscanf(fp, "ly=%lf\n", &ly);
    fscanf(fp, "lz=%lf\n", &lz);
    fscanf(fp, "fet=%s\n", FEname);
    fscanf(fp, "phout=%s\n", phname);
    fscanf(fp, "ds=%lf\n", &ds0);
    fscanf(fp, "Nx=%d\n", &Nx);
    fscanf(fp, "Ny=%d\n", &Ny);
    fclose(fp);

    // double kx[Nx], ky[Ny], kz[Nz];
    wA  = (double*)malloc(sizeof(double) * NxNyNz);
    wB  = (double*)malloc(sizeof(double) * NxNyNz);
    phA = (double*)malloc(sizeof(double) * NxNyNz);
    phB = (double*)malloc(sizeof(double) * NxNyNz);
    eta = (double*)malloc(sizeof(double) * NxNyNz);

    kx         = (double*)malloc(sizeof(double) * Nx);
    ky         = (double*)malloc(sizeof(double) * Ny);
    kz         = (double*)malloc(sizeof(double) * Nz);
    kxyz       = (double*)malloc(sizeof(double) * NxNyNz);
    groupid    = (double*)malloc(sizeof(double) * NxNyNz);
    qbarv      = (double*)malloc(sizeof(double) * NxNyNz);
    Joint_B2A2 = (double*)malloc(sizeof(double) * NxNyNz);
    // qbarv2  = (double*)malloc(sizeof(double) * NxNyNz);
    // wAin=(double *)malloc(sizeof(double)*Nx*Ny*Nz);
    // wBin=(double *)malloc(sizeof(double)*Nx*Ny*Nz);

    /************* x along the cylinder ***************/

    printf("xAB=%lf,Z=%lf\n", hAB, f);
    printf("nx=%d,lx=%lf\n", Nx, lx);

    //	eta[ijk]=(wA[ijk]+wB[ijk]-hAB)/2;
    //	ah=hBC*(hAC+hAB-hBC);
    //  bh=hAC*(hAB+hBC-hAC);
    //  ch=hAB*(hAC+hBC-hAB);
    //  abch=hAB*hAC*hBC;

    //	en1=100.00;
    //  en2=100.00;
    //  en3=100.00;

    //  for(Nlp=1; Nlp<=MaxNlp; Nlp++)
    //	do
    //  {
    //		Nlp=Nlp+1;
    dx = lx / Nx;  // grid spacing in the direction x
    dy = ly / Ny;
    dz = lz / Nz;

    fB  = 1.0 - f;
    fA1 = f / 2.0;
    fA2 = fA1;
    fB2 = fB * tauB2;
    fB1 = (1.0 - f - fB2) / (1.0 + pow(10.0, ksi));
    fB3 = fB1 * pow(10.0, ksi);

    fp = fopen(FEname, "w");
    fprintf(fp, "Nx=%d, Ny=%d, Nz=%d\n", Nx, Ny, Nz);
    // fprintf(fp,"The surface only attract A, and do not repulse B\n");
    fprintf(fp, "hAB=%lf, f=%lf\n", hAB, f);
    fprintf(fp, "fB1=%lf, fA1=%lf, fB2=%lf, fA2=%lf, fB3=%lf\n", fB1, fA1, fB2,
            fA2, fB3);
    fprintf(fp, "lx = %lf, ly = %lf, lz = %lf\n", lx, ly, lz);
    fprintf(fp, "dx=%.6lf, dy=%.6lf, dz=%.6lf\n", dx, dy, dz);
    //	fprintf(fp, "epA = %lf, epB = %lf, epC = %lf\n", epA, epB, epC);
    fclose(fp);

    NsA = ((int)(f / ds0 + 1.0e-8));
    NsB = ((int)(fB / ds0 + 1.0e-8));

    NsA1 = ((int)(fA1 / ds0 + 1.0e-8));
    NsA2 = ((int)(fA2 / ds0 + 1.0e-8));
    NsB1 = ((int)(fB1 / ds0 + 1.0e-8));
    NsB2 = ((int)(fB2 / ds0 + 1.0e-8));
    NsB3 = ((int)(fB3 / ds0 + 1.0e-8));
    //       NsC = ((int)(fC/ds0+1.0e-8));
    printf("NsA = %d, NsB = %d\n", NsA, NsB);
    printf("NsB1 = %d, NsA1 = %d, NsB2 = %d, NsA2 = %d, NsB3 = %d\n", NsB1,
           NsA1, NsB2, NsA2, NsB3);

    //**************************definition of surface field and
    // confinement***********************

    for (i = 0; i < Nx; i++) {
        if (i <= Nx / 2)
            kx[i] = 2 * Pi * i * 1.0 / Nx / dx;
        else
            kx[i] = 2 * Pi * (i - Nx) * 1.0 / dx / Nx;
        kx[i] *= kx[i];
    }

    for (j = 0; j < Ny; j++) {
        if (j <= Ny / 2)
            ky[j] = 2 * Pi * j * 1.0 / Ny / dy;
        else
            ky[j] = 2 * Pi * (j - Ny) * 1.0 / dy / Ny;
        ky[j] *= ky[j];
    }

    for (k = 0; k < Nz; k++) {
        if (k <= Nz / 2)
            kz[k] = 2 * Pi * k * 1.0 / Nz / dz;
        else
            kz[k] = 2 * Pi * (k - Nz) * 1.0 / dz / Nz;
        kz[k] *= kz[k];
    }

    // printf("local_x_start=%d\n",local_x_start);
    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
            for (k = 0; k < Nz; k++) {
                ijk       = (i * Ny + j) * Nz + k;
                kxyz[ijk] = kx[i] + ky[j] + kz[k];
            }

    /***************Initialize wA, wB, wC******************/
    if (in == 0)
        initW(wA, wB);  // BCC
    if (in == 3)
        initW_L(wA, wB);  // Lam
    if (in == 2)
        initW_C(wA, wB);  // Cylinder
    if (in == 5)
        initW_G(wA, wB);  // Gyroid
    else if (in == 1) {
        fp = fopen("pha.dat", "r");
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++)
                for (k = 0; k < Nz; k++) {
                    /*			fscanf(fp,"%lf %lf %lf %lf %lf
                       %lf",&e1,&e2,&e3,&e4,&e5,&e6); ijk=(i*Ny+j)*Nz+k;
                                wA[ijk]=hAB*e2+hAC*e3;
                                wB[ijk]=hAB*e1+hBC*e3;
                                wC[ijk]=hAC*e1+hBC*e2;
                            }
                            fclose(fp);*/
                    fscanf(fp, "%lf %lf %lf %lf", &e1, &e1, &e2, &e3);
                    ijk     = (i * Ny + j) * Nz + k;
                    wA[ijk] = e2;
                    wB[ijk] = e3;
                }
        fclose(fp);
    }
    else if (in == 11) {
        fp = fopen("phin.txt", "r");
        int linshi;
        fscanf(fp, "%d %d %d\n", &linshi, &linshi, &linshi);
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++)
                for (k = 0; k < Nz; k++) {
                    /*			fscanf(fp,"%lf %lf %lf %lf %lf
                       %lf",&e1,&e2,&e3,&e4,&e5,&e6); ijk=(i*Ny+j)*Nz+k;
                                wA[ijk]=hAB*e2+hAC*e3;
                                wB[ijk]=hAB*e1+hBC*e3;
                                wC[ijk]=hAC*e1+hBC*e2;
                            }
                            fclose(fp);*/
                    fscanf(fp, "%lf %lf %lf %lf\n", &e1, &e1, &e2, &e3);
                    ijk     = (i * Ny + j) * Nz + k;
                    wA[ijk] = e2;
                    wB[ijk] = e3;
                }
        fclose(fp);
    }
    else if (in == 12) {
        fp = fopen("phin_with_mask.txt", "r");
        int linshi;
        fscanf(fp, "%d %d %d\n", &linshi, &linshi, &linshi);
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++)
                for (k = 0; k < Nz; k++) {
                    /*			fscanf(fp,"%lf %lf %lf %lf %lf
                       %lf",&e1,&e2,&e3,&e4,&e5,&e6); ijk=(i*Ny+j)*Nz+k;
                                wA[ijk]=hAB*e2+hAC*e3;
                                wB[ijk]=hAB*e1+hBC*e3;
                                wC[ijk]=hAC*e1+hBC*e2;
                            }
                            fclose(fp);*/
                    fscanf(fp, "%lf %lf %lf %lf %lf\n", &e1, &e1, &e2, &e3,
                           &e5);
                    ijk     = (i * Ny + j) * Nz + k;
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
    else if (in == 4)  // gyroid
    {
        fp = fopen("gyroid.dat", "r");
        printf("read from gyroid.dat\n");
        for (i = 0; i < Nx; i++)
            for (j = 0; j < Ny; j++)
                for (k = 0; k < Nz; k++) {
                    fscanf(fp, "%lf %lf %lf %lf", &e1, &e2, &e4, &e4);
                    ijk     = (i * Ny + j) * Nz + k;
                    wA[ijk] = hAB * e2 + 0.40 * (drand48() - 0.50);
                    wB[ijk] = hAB * e1 + 0.40 * (drand48() - 0.50);
                    //          wC[ijk] = hAC*e1+hBC*e2+0.40*(drand48()-0.50);
                }
        fclose(fp);
        printf("done\n");
    }

    /*        else if(in==3){
                    fp=fopen("pha.dat","r");
                    for(i=0;i<Nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
                    {
                        fscanf(fp,"%lf %lf %lf %lf",&e1,&e2,&e4,&e4);
                        ijk=(i*Ny+j)*Nz+k;
                        wA[ijk] = hAB*(e2+e3)/2.0+hAC*(e2+e3)/2.0+
       0.40*(drand48()-0.50); wB[ijk] = hAB*e1+hBC*(e2+e3)/2.0+
       0.40*(drand48()-0.50); wC[ijk] = hAC*e1+hBC*(e2+e3)/2.0+
       0.40*(drand48()-0.50)
                    }
                    fclose(fp);
            }*/

    e1 = freeE(wA, wB, phA, phB, eta);

    free(wA);
    free(wB);
    free(phA);
    free(phB);
    free(eta);
    free(kxyz);
    free(groupid);
    free(qbarv);
    return 1;
}

//********************Output configuration******************************
void write_ph(double* phA, double* phB, double* wA, double* wB)
{
    int   i, j, k;
    long  ijk;
    FILE* fp = fopen(phname, "w");
    //	fprintf(fp,"Nx=%d,Ny=%d,Nz=%d\n",Nx,Ny,Nz);
    //	fprintf(fp,"dx=%lf,dy=%lf,dz=%lf\n",dx,dy,dz);
    fprintf(fp, "%d %d %d\n", Nz, Nx, Ny);
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                ijk = (i * Ny + j) * Nz + k;
                if (in == 1)
                    fprintf(fp, "%lf %lf %lf %lf\n", phA[ijk], phB[ijk],
                            wA[ijk], wB[ijk]);
                else
                    fprintf(fp, "%lf %lf %lf %lf %lf\n", phA[ijk], phB[ijk],
                            wA[ijk], wB[ijk], groupid[ijk]);
            }
            // fprintf(fp,"\n");
        }
        // fprintf(fp,"\n");
    }
    fclose(fp);
    // fp = fopen("para_out.txt", "w");
    // fprintf(fp, "%d,%d,%d,%lf,%lf,%lf", Nx, Ny, Nz, dx, dy, dz);
    // fclose(fp);
}

//*************************************main
// loop****************************************

double freeE(double* wA, double* wB, double* phA, double* phB, double* eta)
{
    int     i, j, k, iter, maxIter;
    long    ijk;
    double  freeEnergy, freeOld, qC;
    double  freeW, freeAB, freeS, freeDiff, freeWsurf;
    double  Sm1, Sm2, wopt, wcmp, beta, psum, fpsum, *psuC;
    double *waDiff, *wbDiff, inCompMax, wa0, wb0;
    double *del, *outs, *wAnew, *wBnew, err;
    int     N_rec;
    FILE*   fp;
    // MPI_Status status;
    psuC   = (double*)malloc(sizeof(double) * NxNyNz);
    waDiff = (double*)malloc(sizeof(double) * NxNyNz);
    wbDiff = (double*)malloc(sizeof(double) * NxNyNz);
    //  wcDiff=(double *)malloc(sizeof(double)*NxNyNz);
    wAnew = (double*)malloc(sizeof(double) * NxNyNz);
    wBnew = (double*)malloc(sizeof(double) * NxNyNz);
    //  wCnew=(double *)malloc(sizeof(double)*NxNyNz);
    del  = (double*)malloc(sizeof(double) * N_hist * 2 * NxNyNz);
    outs = (double*)malloc(sizeof(double) * N_hist * 2 * NxNyNz);

    Sm1        = 2e-9;
    Sm2        = 1e-11;
    maxIter    = MaxIT;
    wopt       = 0.10;
    wcmp       = 0.20;
    beta       = 1.0;
    iter       = 0;
    freeEnergy = 0.0;
    do {
        iter = iter + 1;
        if (iter == 1)
            qC = getConc(phA, phB, 1.0, wA, wB);
        else
            qC = getConcBridge(phA, phB, 1.0, wA, wB);

        double qsum = 0;
        for (ijk = 0; ijk < NxNyNz; ijk++) {
            qsum += qbarv[ijk];
            qbarv[ijk] = qbarv[ijk] * groupid[ijk];
            // qbarv2[ijk] = qbarv2[ijk] * groupid[ijk];
        }
        printf("qsum=%lf\n", qsum);

        freeW     = 0.0;
        freeAB    = 0.0;
        freeS     = 0.0;
        freeWsurf = 0.0;
        inCompMax = 0.0;

        for (ijk = 0; ijk < NxNyNz; ijk++) {
            eta[ijk]  = (wA[ijk] + wB[ijk] - hAB) / 2;
            psum      = 1.0 - phA[ijk] - phB[ijk];
            psuC[ijk] = psum;
            fpsum     = fabs(psum);
            if (fpsum > inCompMax)
                inCompMax = fpsum;
            wAnew[ijk]  = hAB * phB[ijk] + eta[ijk];
            wBnew[ijk]  = hAB * phA[ijk] + eta[ijk];
            waDiff[ijk] = wAnew[ijk] - wA[ijk];
            wbDiff[ijk] = wBnew[ijk] - wB[ijk];
            waDiff[ijk] -= wcmp * psum;
            wbDiff[ijk] -= wcmp * psum;
            freeAB = freeAB + hAB * phA[ijk] * phB[ijk];
            freeW  = freeW -
                    (wA[ijk] * phA[ijk] + wB[ijk] * phB[ijk] + eta[ijk] * psum);
        }
        freeAB /= NxNyNz;
        freeW /= NxNyNz;
        freeWsurf /= NxNyNz;
        freeS = -log(qC);

        freeOld    = freeEnergy;
        freeEnergy = freeAB + freeW + freeS + freeWsurf;
        // FILE *fp1=fopen("result.dat","w");
        // fprintf(fp1,"%lf %10.8e\n",lx,freeEnergy);
        // fclose(fp1);
        // judge the error
        err = error_cal(waDiff, wbDiff, wA, wB);
        // update the history fields, and zero is new fields
        update_flds_hist(waDiff, wbDiff, wAnew, wBnew, del, outs);
        // if achieved some level, anderson-mixing, else simple-mixing
        if (err > 0.01 || iter < 100) {
            for (ijk = 0; ijk < NxNyNz; ijk++) {
                wA[ijk] += wopt * waDiff[ijk];
                wB[ijk] += wopt * wbDiff[ijk];
                //              wC[ijk]+=wopt*wcDiff[ijk];
            }
        }
        else {
            printf("iter  %4d  err  %.8f /***** enter Anderson mixing *****/\n",
                   iter, err);
            N_rec = (iter - 1) < N_hist ? (iter - 1) : N_hist;
            Anderson_mixing(del, outs, N_rec, wA, wB);
        }

        //**** print out the free energy and error results ****

        if (iter == 1 || iter % 10 == 0 || iter >= maxIter) {
            //			FILE *fp=fopen("printout.txt","a");
            // printf("%d\n",iter);
            if (iter == 1)
                fp = fopen("printout_c.txt", "w");
            else
                fp = fopen("printout_c.txt", "a");
            // printf("%10.8e, %10.8e, %10.8e,
            // %10.8e\n",freeEnergy,freeABC,freeW,freeS);
            fprintf(fp, "%d\n", iter);
            fprintf(fp, "%10.8e, %10.8e, %10.8e, %10.8e, %10.8e, %e\n",
                    freeEnergy, freeAB, freeW, freeS, freeWsurf, inCompMax);
            fclose(fp);
        }

        // if(iter%10==0)
        // {
        printf(" %5d : %.8e, %.8e\n", iter, freeEnergy, inCompMax);
        // }
        freeDiff = fabs(freeEnergy - freeOld);
        if (iter == 1 || iter % 10 == 0)
            write_ph(phA, phB, wA, wB);
    } while (iter < maxIter && (inCompMax > Sm1 || freeDiff > Sm2));

    printf("loopratio=%lf bridgeratio=%lf\n", loopratio, bridgeratio);
    fp = fopen("log_bridge.txt", "w");
    fprintf(fp, "%lf\n", bridgeratio);
    fclose(fp);

    fp = fopen("printout_c.txt", "a");
    fprintf(fp, "%d\n", iter);
    fprintf(fp, "%10.8e, %10.8e, %10.8e, %10.8e, %10.8e, %e\n", freeEnergy,
            freeAB, freeW, freeS, freeWsurf, inCompMax);
    fclose(fp);
    write_ph(phA, phB, wA, wB);
    free(psuC);
    free(waDiff);
    free(wbDiff);
    // free(wcDiff);
    free(wAnew);
    free(wBnew);
    // free(wCnew);
    free(del);
    free(outs);
    return freeDiff;
}

double getConc(double* phlA, double* phlB, double phs0, double* wA, double* wB)
{
    int     i, j, k, iz;
    long    ijk, ijkiz;
    double *qA, *qcA, *qB, *qcB;
    double *qA1, *qcA1, *qA2, *qcA2, *qB1, *qcB1, *qB2, *qcB2, *qB3, *qcB3;
    double  ffl, fflA, fflB, dzA, dzB, *qInt, qtmp;
    double  fflA1, fflA2, fflB1, fflB2, fflB3;
    double  dzA1, dzA2, dzB1, dzB2, dzB3;
    // MPI_Status status;

    // qA=(double *)malloc(sizeof(double)*NxNyNz*(NsA+1));
    // qcA=(double *)malloc(sizeof(double)*NxNyNz*(NsA+1));
    // qB=(double *)malloc(sizeof(double)*NxNyNz*(NsB+1));
    // qcB=(double *)malloc(sizeof(double)*NxNyNz*(NsB+1));

    qB1  = (double*)malloc(sizeof(double) * NxNyNz * (NsB1 + 1));
    qcB1 = (double*)malloc(sizeof(double) * NxNyNz * (NsB1 + 1));
    qA1  = (double*)malloc(sizeof(double) * NxNyNz * (NsA1 + 1));
    qcA1 = (double*)malloc(sizeof(double) * NxNyNz * (NsA1 + 1));
    qB2  = (double*)malloc(sizeof(double) * NxNyNz * (NsB2 + 1));
    qcB2 = (double*)malloc(sizeof(double) * NxNyNz * (NsB2 + 1));
    qA2  = (double*)malloc(sizeof(double) * NxNyNz * (NsA2 + 1));
    qcA2 = (double*)malloc(sizeof(double) * NxNyNz * (NsA2 + 1));
    qB3  = (double*)malloc(sizeof(double) * NxNyNz * (NsB3 + 1));
    qcB3 = (double*)malloc(sizeof(double) * NxNyNz * (NsB3 + 1));
    //  qC=(double *)malloc(sizeof(double)*NxNyNz*(NsC+1));
    //  qcC=(double *)malloc(sizeof(double)*NxNyNz*(NsC+1));
    qInt = (double*)malloc(sizeof(double) * NxNyNz);
    // for(i=0;i<local_nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
    for (ijk = 0; ijk < NxNyNz; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk] = 1.0;
    }

    sovDifFft(qB1, wB, qInt, fB1, NsB1, 1);  // 0 to fB1 for qB1

    sovDifFft(qcB3, wB, qInt, fB3, NsB3, -1);  // 1 to fB3 for qcB3

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qB1[ijk * (NsB1 + 1) + NsB1];
    }
    sovDifFft(qA1, wA, qInt, fA1, NsA1, 1);  // fB1 to fA1 for qA1

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk]  = qA1[ijk * (NsA1 + 1) + NsA1];
        qbarv[ijk] = qA1[ijk * (NsA1 + 1) + NsA1];
    }
    sovDifFft(qB2, wB, qInt, fB2, NsB2, 1);  // fA1 to fB2 for qB2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qB2[ijk * (NsB2 + 1) + NsB2];
    }
    sovDifFft(qA2, wA, qInt, fA2, NsA2, 1);  // fB2 to fA2 for qA2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qA2[ijk * (NsA2 + 1) + NsA2];
    }
    sovDifFft(qB3, wB, qInt, fB3, NsB3, 1);  // fA2 to fB3 for qB3

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcB3[ijk * (NsB3 + 1)];
    }
    sovDifFft(qcA2, wA, qInt, fA2, NsA2, -1);  // fB3 to fA2 for qcA2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcA2[ijk * (NsA2 + 1)];
    }
    sovDifFft(qcB2, wB, qInt, fB2, NsB2, -1);  // fA2 to fB2 for qcB2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcB2[ijk * (NsB2 + 1)];
    }
    sovDifFft(qcA1, wA, qInt, fA1, NsA1, -1);  // fB2 to fA1 for qcA1

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcA1[ijk * (NsA1 + 1)];
    }
    sovDifFft(qcB1, wB, qInt, fB1, NsB1, -1);  // fA1 to fB1 for qcB1

    ql    = 0.0;
    ZDIMM = NsB1 + 1;
    // for(i=0;i<Nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
    for (ijk = 0; ijk < NxNyNz; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        // printf("%.10lf\n", qcB1[ijk * ZDIMM]);
        ql += qcB1[ijk * ZDIMM];
    }

    ql /= NxNyNz;

    ffl  = phs0 / ql;
    dzA1 = fA1 / NsA1;
    dzA2 = fA2 / NsA2;
    dzB1 = fB1 / NsB1;
    dzB2 = fB2 / NsB2;
    dzB3 = fB3 / NsB3;

    fflA1 = dzA1 * ffl;
    fflA2 = dzA2 * ffl;
    fflB1 = dzB1 * ffl;
    fflB2 = dzB2 * ffl;
    fflB3 = dzB3 * ffl;

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        phlA[ijk] = 0.0;
        phlB[ijk] = 0.0;

        ZDIMM = NsA1 + 1;
        qtmp  = 0.0;
        for (iz = 0; iz <= NsA1; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsA1)
                qtmp += (0.50 * qA1[ijkiz] * qcA1[ijkiz]);
            else
                qtmp += (qA1[ijkiz] * qcA1[ijkiz]);
        }
        phlA[ijk] += qtmp * fflA1;

        ZDIMM = NsA2 + 1;
        qtmp  = 0.0;
        for (iz = 0; iz <= NsA2; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsA2)
                qtmp += (0.50 * qA2[ijkiz] * qcA2[ijkiz]);
            else
                qtmp += (qA2[ijkiz] * qcA2[ijkiz]);
        }
        phlA[ijk] += qtmp * fflA2;

        ZDIMM = NsB1 + 1;
        qtmp  = 0.0;
        for (iz = 0; iz <= NsB1; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsB1)
                qtmp += (0.50 * qB1[ijkiz] * qcB1[ijkiz]);
            else
                qtmp += (qB1[ijkiz] * qcB1[ijkiz]);
        }
        phlB[ijk] += qtmp * fflB1;

        ZDIMM = NsB2 + 1;
        qtmp  = 0.0;
        for (iz = 0; iz <= NsB2; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsB2)
                qtmp += (0.50 * qB2[ijkiz] * qcB2[ijkiz]);
            else
                qtmp += (qB2[ijkiz] * qcB2[ijkiz]);
        }
        phlB[ijk] += qtmp * fflB2;

        ZDIMM = NsB3 + 1;
        qtmp  = 0.0;
        for (iz = 0; iz <= NsB3; iz++) {
            ijkiz = ijk * ZDIMM + iz;
            if (iz == 0 || iz == NsB3)
                qtmp += (0.50 * qB3[ijkiz] * qcB3[ijkiz]);
            else
                qtmp += (qB3[ijkiz] * qcB3[ijkiz]);
        }
        phlB[ijk] += qtmp * fflB3;
    }

    free(qA);
    free(qB);
    free(qcA);
    free(qcB);
    free(qA1);
    free(qcA1);
    free(qA2);
    free(qcA2);
    free(qB1);
    free(qcB1);
    free(qB2);
    free(qcB2);
    free(qB3);
    free(qcB3);
    free(qInt);
    return ql;
}

double
getConcBridge(double* phlA, double* phlB, double phs0, double* wA, double* wB)
{
    int     i, j, k, iz;
    long    ijk, ijkiz;
    double *qA, *qcA, *qB, *qcB;
    double *qA1, *qcA1, *qA2, *qcA2, *qB1, *qcB1, *qB2, *qcB2, *qB3, *qcB3;
    double  ffl, fflA, fflB, dzA, dzB, *qInt, qtmp;
    double  fflA1, fflA2, fflB1, fflB2, fflB3;
    double  dzA1, dzA2, dzB1, dzB2, dzB3;
    double* qIntb;

    qB1   = (double*)malloc(sizeof(double) * NxNyNz * (NsB1 + 1));
    qcB1  = (double*)malloc(sizeof(double) * NxNyNz * (NsB1 + 1));
    qA1   = (double*)malloc(sizeof(double) * NxNyNz * (NsA1 + 1));
    qcA1  = (double*)malloc(sizeof(double) * NxNyNz * (NsA1 + 1));
    qB2   = (double*)malloc(sizeof(double) * NxNyNz * (NsB2 + 1));
    qcB2  = (double*)malloc(sizeof(double) * NxNyNz * (NsB2 + 1));
    qA2   = (double*)malloc(sizeof(double) * NxNyNz * (NsA2 + 1));
    qcA2  = (double*)malloc(sizeof(double) * NxNyNz * (NsA2 + 1));
    qB3   = (double*)malloc(sizeof(double) * NxNyNz * (NsB3 + 1));
    qcB3  = (double*)malloc(sizeof(double) * NxNyNz * (NsB3 + 1));
    qInt  = (double*)malloc(sizeof(double) * NxNyNz);
    qIntb = (double*)malloc(sizeof(double) * NxNyNz);

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        // ijk=(i*Ny+j)*Nz+k;
        qInt[ijk]  = 1.0;
        qIntb[ijk] = qbarv[ijk];
    }

    // for (ijk = 0; ijk < NxNyNz; ijk++) {
    //     qIntb[ijk] = qbarv[ijk];
    //     // qIntb2[ijk] = qbarv2[ijk];
    // }

    // sovDifFft(qB1, wB, qInt, fB1, NsB1, 1);  // 0 to fB1 for qB1
    sovDifFft(qA1, wA, qIntb, fA1, NsA1, -1);
    sovDifFft(qcB3, wB, qInt, fB3, NsB3, -1);  // 1 to fB3 for qcB3

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qA1[ijk * (NsA1 + 1)];
    }
    sovDifFft(qB1, wB, qInt, fB1, NsB1, -1);

    // for (ijk = 0; ijk < NxNyNz; ijk++) {
    //     qInt[ijk] = qB1[ijk * (NsB1 + 1) + NsB1];
    // }
    // sovDifFft(qA1, wA, qInt, fA1, NsA1, 1);  // fB1 to fA1 for qA1

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qA1[ijk * (NsA1 + 1) + NsA1];
    }
    sovDifFft(qB2, wB, qInt, fB2, NsB2, 1);  // fA1 to fB2 for qB2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qB2[ijk * (NsB2 + 1) + NsB2];
    }
    sovDifFft(qA2, wA, qInt, fA2, NsA2, 1);  // fB2 to fA2 for qA2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qA2[ijk * (NsA2 + 1) + NsA2];
    }
    sovDifFft(qB3, wB, qInt, fB3, NsB3, 1);  // fA2 to fB3 for qB3

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcB3[ijk * (NsB3 + 1)];
    }
    sovDifFft(qcA2, wA, qInt, fA2, NsA2, -1);  // fB3 to fA2 for qcA2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcA2[ijk * (NsA2 + 1)];
    }
    sovDifFft(qcB2, wB, qInt, fB2, NsB2, -1);  // fA2 to fB2 for qcB2

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcB2[ijk * (NsB2 + 1)];
    }
    sovDifFft(qcA1, wA, qInt, fA1, NsA1, -1);  // fB2 to fA1 for qcA1

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        qInt[ijk] = qcA1[ijk * (NsA1 + 1)];
    }
    sovDifFft(qcB1, wB, qInt, fB1, NsB1, -1);  // fA1 to fB1 for qcB1

    // ql    = 0.0;
    // ZDIMM = NsB1 + 1;
    // // for(i=0;i<Nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
    // for (ijk = 0; ijk < NxNyNz; ijk++) {
    //     // ijk=(i*Ny+j)*Nz+k;
    //     // printf("%.10lf\n", qcB1[ijk * ZDIMM]);
    //     ql += qcB1[ijk * ZDIMM];
    // }

    // ql /= NxNyNz;

    ffl  = phs0 / ql;
    dzA1 = fA1 / NsA1;
    dzA2 = fA2 / NsA2;
    dzB1 = fB1 / NsB1;
    dzB2 = fB2 / NsB2;
    dzB3 = fB3 / NsB3;

    fflA1 = dzA1 * ffl;
    fflA2 = dzA2 * ffl;
    fflB1 = dzB1 * ffl;
    fflB2 = dzB2 * ffl;
    fflB3 = dzB3 * ffl;

    // dzA = f / NsA;
    // dzB = fB / NsB;

    // fflA = dzA * ffl;
    // fflB = dzB * ffl;

    loopratio = 0;
    // loopratio_test = 0;

    fflloop = 1.0 / ql / vcell;
    printf("ql=%.10lf, vcell=%lf, fflloop=%lf\n", ql, vcell, fflloop);

    // int ZDIMM_B2_s, ZDIMM_B2_e;

    FILE* fp;
    fp = fopen("Joint_B2A2.txt", "w");
    fprintf(fp, "%d %d %d\n", Nz, Nx, Ny);

    ZDIMM = NsA2 + 1;
    for (ijk = 0; ijk < NxNyNz; ijk++) {
        loopratio +=
            qA2[ijk * ZDIMM] * qcA2[ijk * ZDIMM] * fflloop * groupid[ijk];
        Joint_B2A2[ijk] = qA2[ijk * ZDIMM] * qcA2[ijk * ZDIMM] * fflA2;
        fprintf(fp, "%f\n", Joint_B2A2[ijk]);
    }
    fclose(fp);

    bridgeratio = 1 - loopratio;
    printf("loopratio=%lf bridgeratio=%lf\n", loopratio, bridgeratio);

    free(qA);
    free(qB);
    free(qcA);
    free(qcB);
    free(qA1);
    free(qcA1);
    free(qA2);
    free(qcA2);
    free(qB1);
    free(qcB1);
    free(qB2);
    free(qcB2);
    free(qB3);
    free(qcB3);
    free(qInt);
    free(qIntb);
    return ql;
}

void sovDifFft(double* g, double* w, double* qInt, double z, int ns, int sign)
{
    int           i, j, k, iz;
    unsigned long ijk, ijkr;
    double        dzc, *wdz;
    double *      kxyzdz, dzc2;
    double*       in;
    fftw_complex* out;
    fftw_plan     p_forward, p_backward;

    wdz    = (double*)malloc(sizeof(double) * NxNyNz);
    kxyzdz = (double*)malloc(sizeof(double) * NxNyNz);
    in     = (double*)malloc(sizeof(double) * NxNyNz);

    out   = (fftw_complex*)malloc(sizeof(fftw_complex) * NxNyNz1);
    dzc   = z / ns;
    dzc2  = 0.50 * dzc;
    ZDIMM = ns + 1;
    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
            for (k = 0; k < Nz; k++)
            // for(ijk=0;ijk<total_local_size;ijk++)
            {
                ijk         = (i * Ny + j) * Nz + k;
                kxyzdz[ijk] = exp(-dzc * kxyz[ijk]);
                wdz[ijk]    = exp(-w[ijk] * dzc2);
            }
    p_forward  = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, in, out, FFTW_ESTIMATE);
    p_backward = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, out, in, FFTW_ESTIMATE);
    if (sign == 1) {
        // for(i=0;i<local_nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
        for (ijk = 0; ijk < NxNyNz; ijk++) {
            // ijk=(i*Ny+j)*Nz+k;
            g[ijk * ZDIMM] = qInt[ijk];
        }
        for (iz = 1; iz <= ns; iz++) {
            // for(i=0;i<Nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
            for (ijk = 0; ijk < NxNyNz; ijk++) {
                in[ijk] = g[ijk * ZDIMM + iz - 1] * wdz[ijk];
            }

            fftw_execute(p_forward);

            for (i = 0; i < Nx; i++)
                for (j = 0; j < Ny; j++)
                    for (k = 0; k < Nzh1; k++)
                    // for(ijk=0;ijk<total_local_size;ijk++)
                    {
                        ijk  = (i * Ny + j) * Nzh1 + k;
                        ijkr = (i * Ny + j) * Nz + k;
                        out[ijk][0] *=
                            kxyzdz[ijkr];  // out[].re or .im for fftw2
                        out[ijk][1] *=
                            kxyzdz[ijkr];  // out[][0] or [1] for fftw3
                    }

            fftw_execute(p_backward);

            // for(i=0;i<Nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
            for (ijk = 0; ijk < NxNyNz; ijk++) {
                // ijk=(i*Ny+j)*Nz2+k;
                // ijkr=(i*Ny+j)*Nz+k;
                g[ijk * ZDIMM + iz] = in[ijk] * wdz[ijk] / NxNyNz;
            }
        }
    }
    else {
        // for(i=0;i<local_nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
        for (ijk = 0; ijk < NxNyNz; ijk++) {
            // ijk=(i*Ny+j)*Nz+k;
            g[ijk * ZDIMM + ns] = qInt[ijk];
        }
        for (iz = ns - 1; iz >= 0; iz--) {
            for (ijk = 0; ijk < NxNyNz; ijk++) {
                in[ijk] = g[ijk * ZDIMM + iz + 1] * wdz[ijk];
            }

            fftw_execute(p_forward);

            for (i = 0; i < Nx; i++)
                for (j = 0; j < Ny; j++)
                    for (k = 0; k < Nzh1; k++)
                    // for(ijk=0;ijk<total_local_size;ijk++)
                    {
                        ijk  = (i * Ny + j) * Nzh1 + k;
                        ijkr = (i * Ny + j) * Nz + k;
                        out[ijk][0] *= kxyzdz[ijkr];
                        out[ijk][1] *= kxyzdz[ijkr];
                    }

            fftw_execute(p_backward);

            // for(i=0;i<Nx;i++)for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
            for (ijk = 0; ijk < NxNyNz; ijk++) {
                // ijk=(i*Ny+j)*Nz2+k;
                // ijkr=(i*Ny+j)*Nz+k;
                g[ijk * ZDIMM + iz] = in[ijk] * wdz[ijk] / NxNyNz;
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
    for (ijk = 0; ijk < NxNyNz; ijk++) {
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
        for (ijk = 0; ijk < NxNyNz; ijk++) {
            Del(0, ijk, j) = Del(0, ijk, j - 1);
            Del(1, ijk, j) = Del(1, ijk, j - 1);
            //              Del(2, ijk, j) = Del(2, ijk, j-1);
            Outs(0, ijk, j) = Outs(0, ijk, j - 1);
            Outs(1, ijk, j) = Outs(1, ijk, j - 1);
            //              Outs(2, ijk, j) = Outs(2, ijk, j-1);
        }
        //          printf("outs[%d] = %lf\n", j, Outs(1, 10, j));
    }

    for (ijk = 0; ijk < NxNyNz; ijk++) {
        Del(0, ijk, 0) = waDiff[ijk];
        Del(1, ijk, 0) = wbDiff[ijk];
        //          Del(2, ijk, 0) = wcDiff[ijk];
        Outs(0, ijk, 0) = wAnew[ijk];
        Outs(1, ijk, 0) = wBnew[ijk];
        //          Outs(2, ijk, 0) = wCnew[ijk];
    }
    //	printf("outs[0] = %lf\n", Outs(1,10,j));
    //	getchar();
}

/*********************************************************************/
/*
  Anderson mixing [O(Nx)] CHECKED
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
        for (ijk = 0; ijk < NxNyNz; ijk++) {
            V(n) += (Del(0, ijk, 0) - Del(0, ijk, n)) * Del(0, ijk, 0);
            V(n) += (Del(1, ijk, 0) - Del(1, ijk, n)) * Del(1, ijk, 0);
            //          V(n) += ( Del(2,ijk,0) - Del(2,ijk,n) )*Del(2,ijk,0);
        }

        for (m = n; m < N_rec; m++) {
            U(n, m) = 0.0;
            for (ijk = 0; ijk < NxNyNz; ijk++) {
                U(n, m) += (Del(0, ijk, 0) - Del(0, ijk, n)) *
                           (Del(0, ijk, 0) - Del(0, ijk, m));
                U(n, m) += (Del(1, ijk, 0) - Del(1, ijk, n)) *
                           (Del(1, ijk, 0) - Del(1, ijk, m));
                //          U(n,m) += ( Del(2,ijk,0) - Del(2,ijk,n) )*(
                //          Del(2,ijk,0) - Del(2,ijk,m) );
            }
            U(m, n) = U(n, m);
        }
    }

    /* Calculate A - uses GNU LU decomposition for U A = V */
    uGnu = gsl_matrix_view_array(up, N_rec - 1, N_rec - 1);
    vGnu = gsl_vector_view_array(vp, N_rec - 1);
    aGnu = gsl_vector_view_array(ap, N_rec - 1);
    p    = gsl_permutation_alloc(N_rec - 1);
    gsl_linalg_LU_decomp(&uGnu.matrix, p, &s);
    gsl_linalg_LU_solve(&uGnu.matrix, p, &vGnu.vector, &aGnu.vector);
    gsl_permutation_free(p);

    /* Update omega */
    for (ijk = 0; ijk < NxNyNz; ijk++) {
        wA[ijk] = Outs(0, ijk, 0);
        wB[ijk] = Outs(1, ijk, 0);
        //      wC[ijk] = Outs(2,ijk,0);
        for (n = 1; n < N_rec; n++) {
            wA[ijk] += A(n) * (Outs(0, ijk, n) - Outs(0, ijk, 0));
            wB[ijk] += A(n) * (Outs(1, ijk, n) - Outs(1, ijk, 0));
            //          wC[ijk] += A(n)*(Outs(2,ijk,n) - Outs(2,ijk,0));
        }
    }
    free(ap);
    free(vp);
    free(up);
}

void initW(double* wA, double* wB)
{
    int    i, j, k, nc, tag;
    long   ijk;
    double xij, yij, zij;
    double xc[9], yc[9], zc[9];
    double xi, yj, zk, rij, r0, r1;
    double phat, phbt;

    xc[0] = 0.0;
    yc[0] = 0.0;
    zc[0] = 0.0;
    xc[1] = 0.0;
    yc[1] = ly;
    zc[1] = 0.0;
    xc[2] = lx;
    yc[2] = 0.0;
    zc[2] = 0.0;
    xc[3] = 0.0;
    yc[3] = 0.0;
    zc[3] = lz;
    xc[4] = lx;
    yc[4] = ly;
    zc[4] = 0.0;
    xc[5] = lx;
    yc[5] = 0.0;
    zc[5] = lz;
    xc[6] = 0.0;
    yc[6] = ly;
    zc[6] = lz;
    xc[7] = lx;
    yc[7] = ly;
    zc[7] = lz;
    xc[8] = lx / 2.0;
    yc[8] = ly / 2.0;
    zc[8] = lz / 2.0;

    r0 = pow((f * lx * ly * lz / (4.0 * Pi / 3.0)), 1.0 / 3.0);
    //  r1 = pow((fC*lx*ly*lz/(4.0*Pi/3.0)), 1.0/3.0);
    for (i = 0; i < Nx; i++) {
        xi = i * dx;
        for (j = 0; j < Ny; j++) {
            yj = j * dy;
            for (k = 0; k < Nz; k++) {
                //				xi = i*dx;
                //              yj = j*dy;
                zk = k * dz;
                //              phat = 0.0; phbt = 1.0; phct = 0.0;
                tag = 0;
                for (nc = 0; nc < 9; nc++) {
                    xij = xi - xc[nc];
                    yij = yj - yc[nc];
                    zij = zk - zc[nc];
                    rij = xij * xij + yij * yij + zij * zij;
                    rij = sqrt(rij);
                    //                  rcij =
                    //                  (xi-lx/2)*(xi-lx/2)+(yj-ly/2)*(yj-ly/2)+(zk-lz/2)*(zk-lz/2);
                    //                  rcij=sqrt(rcij);
                    if (rij < r0)
                        tag = 1;
                }
                phat = 0.0;
                phbt = 1.0;
                if (tag) {
                    phat = 1.0;
                    phbt = 0.0;
                    //                  else if  phat = 0.0; phbt = 1.0; phct =
                    //                  0.0;
                }
                ijk     = (i * Ny + j) * Nz + k;
                wA[ijk] = hAB * phbt + 0.040 * (drand48() - 0.5);
                wB[ijk] = hAB * phat + 0.040 * (drand48() - 0.5);
                //              wC[ijk]=hAC*phat+hBC*phbt+0.040*(drand48()-0.5);
            }
        }
    }
}

void initW_C(double* wA, double* wB)
{
    int    i, j, k, nc, tag;
    long   ijk;
    double xij, yij, zij;
    double xc[5], yc[5], zc[5];
    double xi, yj, zk, rij, r0, r1;
    double phat, phbt;

    xc[0] = 0.0;
    yc[0] = 0.0;
    xc[1] = 0.0;
    yc[1] = ly;
    xc[2] = lx;
    yc[2] = 0.0;
    xc[3] = lx;
    yc[3] = ly;
    xc[4] = lx / 2;
    yc[4] = ly / 2;

    r0 = pow((f * lx * ly / Pi), 1.0 / 2);
    //	r1 = pow((fC*lx*ly/Pi), 1.0/2);
    for (i = 0; i < Nx; i++) {
        xi = i * dx;
        for (j = 0; j < Ny; j++) {
            yj = j * dy;
            for (k = 0; k < Nz; k++) {
                zk  = k * dz;
                tag = 0;
                for (nc = 0; nc <= 3; nc++) {
                    xij = xi - xc[nc];
                    yij = yj - yc[nc];
                    rij = xij * xij + yij * yij;
                    rij = sqrt(rij);
                    if (rij < r0)
                        tag = 1;
                }
                phat = 0.0;
                phbt = 1.0;
                if (tag) {
                    phat = 1.0;
                    phbt = 0.0;
                }
                tag = 0;
                for (nc = 4; nc < 5; nc++) {
                    xij = xi - xc[nc];
                    yij = yj - yc[nc];
                    rij = xij * xij + yij * yij;
                    rij = sqrt(rij);
                    if (rij < r1)
                        tag = 1;
                }
                if (tag) {
                    phat = 1.0;
                    phbt = 0.0;
                }
                ijk     = (i * Ny + j) * Nz + k;
                wA[ijk] = hAB * phbt + 0.040 * (drand48() - 0.5);
                wB[ijk] = hAB * phat + 0.040 * (drand48() - 0.5);
                //              wC[ijk]=hAC*phat+hBC*phbt+0.040*(drand48()-0.5);
            }
        }
    }
}

void initW_L(double* wA, double* wB)
{
    int    i, j, k, tag, nc;
    long   ijk;
    double xij, yij, zij, ri, r0, Sum;
    double xi, yj, zk;
    double xc[1];
    double phat, phbt;

    xc[0] = lx * (1 - f) / 2;
    xc[1] = lx * (1 + f) / 2;
    r0    = lx * f;
    //  Sum=0.0;
    for (i = 0; i < Nx; i++) {
        xi = i * dx;
        for (j = 0; j < Ny; j++) {
            yj = j * dy;
            for (k = 0; k < Nz; k++) {
                zk  = k * dz;
                tag = 0;
                Sum = 0;
                for (nc = 0; nc <= 1; nc++) {
                    xi = abs(xi - xc[nc]);
                    Sum += xi;
                    if (Sum < r0)
                        tag = 1;
                }
                phat = 0.0;
                phbt = 1.0;
                if (tag) {
                    phat = 1.0;
                    phbt = 0.0;
                }
                ijk     = (i * Ny + j) * Nz + k;
                wA[ijk] = hAB * phbt + 0.040 * (drand48() - 0.5);
                wB[ijk] = hAB * phat + 0.040 * (drand48() - 0.5);
            }
        }
    }
}

void initW_G(double* wA, double* wB)
{
    int    i, j, k;
    long   ijk;
    double phat, phbt, v;
    for (i = 0; i < Nx; i++)
        for (j = 0; j < Ny; j++)
            for (k = 0; k < Nz; k++) {
                ijk = (i * Ny + j) * Nz + k;
                v   = sin(2 * Pi * i / Nx) * cos(2 * Pi * j / Ny);
                v += sin(2 * Pi * k / Nz) * cos(2 * Pi * i / Nx);
                v += sin(2 * Pi * j / Ny) * cos(2 * Pi * k / Nz);
                v    = fabs(v);
                phat = 0.0;
                phbt = 1.0;
                if (v > 1.5 * (1 - f)) {
                    phat = 1.0;
                    phbt = 0.0;
                }
                wA[ijk] = hAB * phbt + 0.040 * (drand48() - 0.5);
                wB[ijk] = hAB * phat + 0.040 * (drand48() - 0.5);
            }
}
// Ref:Paul J.F.Gandy.2001.Nodal surface approximations to the P, G, D and I-WP
// triply periodic minimal surfaces.
