//
//  main.cpp
//  openmpi
//
//  Created by Антон Кудряшов on 05/06/16.
//  Copyright © 2016 Антон Кудряшов. All rights reserved.
//

#include <iostream>

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include "bounds.h"

#define mD 6

//interface
void TDMAParallel(double* a, double* b, double* c, double* d, double* array, int N,
                  boundsConditions bounds0, double* a_0, double* b_0, double* c_0, double* d_0);

void TDMA(double* a, double* b, double* c, double* d, double* array, int N, boundsConditions bounds);

//support functions
double f(double x)
{
    return x*x*x*x/20000 + sin(x);
}
double spline(double v, double* x, double* y, double* u)
{
    const int N = mD;
    if (v > x[N]) {
        return y[N];
    }
    int i = 0;
    for (int j = 1; j <= N; j++) {
        if (x[j] >= v) {
            i = j;
            break;
        }
    }
    const double l = x[i] - v;
    const double r = v - x[i - 1];
    const double h = x[i] - x[i - 1];
    
    return y[i-1]*l/h + y[i]*r/h
    + u[i-1]*(pow(l, 3) - pow(h, 2)*l)/(6*h)
    + u[i]*(pow(r, 3) - pow(h, 2)*r)/(6*h);
}

void printA(double* a, int size)
{
    for (int i=0; i<size; i++) {
        printf("= %.5f\n", a[i]);
    }
}

int main(int argc, const char * argv[])
{
    int size, rank;
    //MPI Initialization
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //Fill interpolation data for each processor
    int m = mD;// N/size
    int N = size*m;
    
    const double L = -4;
    const double R = -6;
    
    double* A0 = nullptr;
    double* B0 = nullptr;
    double* C0 = nullptr;
    double* D0 = nullptr;
    double* a = new double[m + 1];
    double* b = new double[m + 1];
    double* c = new double[m + 1];
    double* d = new double[m + 1];
    double* result = new double[m + 1];
    
    double* x = new double[m + 1];
    double* y = new double[m + 1];
    double* h = new double[m + 1];
    for (int i=0; i <= m; i++) {
        x[i] = 2*i + rank*m;
        double differ = i > 0 && i < m ? arc4random()%2 : 0;
        y[i] = f(x[i]);
    }
    for (int i=1; i <= m; i++) {
        h[i] = x[i] - x[i-1];
    }
    
    boundValue leftBound0;
    leftBound0.type = boundsNormal;
    leftBound0.k = -0.5;
    leftBound0.r = 3.*((y[1]-y[0])/h[1] - L)/h[1];
    
    boundsConditions bounds0;
    bounds0.left = leftBound0;
    
    for (int i=1; i < m; i++) {
        a[i] = h[i]/6.;
        c[i] = (h[i]+h[i+1])/3.;
        b[i] = h[i+1]/6.;
        d[i] = (y[i+1] - y[i])/h[i+1] - (y[i] - y[i-1])/h[i];
    }
    double cache[5];
    if(rank == 0) {
        A0 = new double[size + 1];
        B0 = new double[size + 1];
        C0 = new double[size + 1];
        D0 = new double[size + 1];
        double h_prev = h[m];
        double y_prev = y[m-1];
        for (int i=1; i < size; i++) {
            MPI_Status status;
            MPI_Recv(cache, 5, MPI_DOUBLE, i, 200, MPI_COMM_WORLD, &status);
            A0[i] = h_prev/6.;
            C0[i] = (h_prev+cache[0])/3.;
            B0[i] = cache[0]/6.;
            D0[i] = (cache[2] - cache[1])/cache[0] - (cache[1] - y_prev)/h_prev;
            y_prev = cache[3];
            h_prev = cache[4];
        }
        double Rbuf;
        MPI_Status status;
        MPI_Recv(&Rbuf, 1, MPI_DOUBLE, size - 1, 201, MPI_COMM_WORLD, &status);
        boundValue rightBound0;
        rightBound0.type = boundsNormal;
        rightBound0.k = -0.5;
        rightBound0.r = Rbuf;
        
        bounds0.right = rightBound0;
    }
    else {
        cache[0] = h[1];
        cache[1] = y[0];
        cache[2] = y[1];
        cache[3] = y[m-1];
        cache[4] = h[m];
        MPI_Send(cache, 5, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
        if (rank == size - 1) {
            double Rbuf = 3.*(R - (y[m]-y[m-1])/h[m])/h[m];
            MPI_Send(&Rbuf, 1, MPI_DOUBLE, 0, 201, MPI_COMM_WORLD);
        }
    }
    // waiting for all processors
    MPI_Barrier(MPI_COMM_WORLD);
    
    // method execution
    TDMAParallel(a, b, c, d, result, m + 1, bounds0, A0, B0, C0, D0);
    
    // finalize method and print output data in mathematica nb-file
    double eps = 0.00001;
    //printf("%d L error: %f\tR error: %f\n", rank, L-(spline(x[0] + eps, x, y, result) - spline(x[0], x, y, result))/eps, R-(spline(x[m], x, y, result) - spline(x[m]-eps, x, y, result))/eps);
    bool print = true;
    if (print) {
        if (rank == 0) {
            printf("list = {\n");
            for (double v=x[0]; v<=x[m];) {
                if (v > 0) {
                    printf(",\n");
                }
                printf("\t{%.5f,\t%.5f}", v, spline(v, x, y, result));
                v += 0.01;
            }
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(result, m + 1, MPI_DOUBLE, i, 99, MPI_COMM_WORLD, &status);
                MPI_Recv(x, m + 1, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, &status);
                MPI_Recv(y, m + 1, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, &status);
                for (double v=x[0]; v<=x[m];) {
                    printf(",\n");
                    printf("\t{%.5f,\t%.5f}", v, spline(v, x, y, result));
                    v += 0.01;
                }
            }
            printf("}\n Show[ListLinePlot[list, PlotStyle -> Black], Plot[x*x*x*x/20000 + Sin[x], {x, 0, %f}]]\n", x[m]);
        }
        else {
            MPI_Send(result, m + 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
            MPI_Send(x, m + 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
            MPI_Send(y, m + 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
        }
    }
    
    if (rank == 0) {
        delete [] A0;
        delete [] B0;
        delete [] C0;
        delete [] D0;
    }
    delete [] x;
    delete [] y;
    delete [] h;
    
    delete [] a;
    delete [] b;
    delete [] c;
    delete [] d;
    delete [] result;
    MPI_Finalize();
    return 0;
}

void TDMAParallel(double* a, double* b, double* c, double* d, double* array, int N,
                  boundsConditions bounds0, double* a_0, double* b_0, double* c_0, double* d_0)
{
    if (N < 3) {
        printf("\n\n!!! N less than 3\n\n");
    }
    double* dZero = new double[N];
    double* u = new double[N];
    double* v = new double[N];
    double* w = new double[N];
    for (int i =0; i < N; i++) {
        dZero[i] = 0;
    }
    boundValue leftBound, rightBound;
    //split TDMA into 3 TDMA systems
    boundsConditions boundsW;
    leftBound.r = 0;leftBound.k = 0;
    rightBound.r = 0;rightBound.k = 0;
    boundsW.left = leftBound;
    boundsW.right = rightBound;
    TDMA(a, b, c, d, w, N, boundsW);
    
    boundsConditions boundsU;
    leftBound.r = 1;leftBound.k = 0;
    rightBound.r = 0;rightBound.k = 0;
    boundsU.left = leftBound;
    boundsU.right = rightBound;
    TDMA(a, b, c, dZero, u, N, boundsU);
    
    boundsConditions boundsV;
    leftBound.r = 0;leftBound.k = 0;
    rightBound.r = 1;rightBound.k = 0;
    boundsV.left = leftBound;
    boundsV.right = rightBound;
    TDMA(a, b, c, dZero, v, N, boundsV);
    
    int size, rank;
    MPI_Comm commutator = MPI_COMM_WORLD;//change if needed
    MPI_Comm_size(commutator, &size);
    MPI_Comm_rank(commutator, &rank);
    
    double buff[6];
    
    double u_Nm, v_Nm, w_Nm, u_1, v_1, w_1;
    buff[0] = u_1  = u[1];
    buff[1] = u_Nm = u[N-2];
    buff[2] = v_1  = v[1];
    buff[3] = v_Nm = v[N-2];
    buff[4] = w_1  = w[1];
    buff[5] = w_Nm = w[N-2];
    
    const int tagInput = 5;
    const int tagOutput = 6;
    double z_buff[2];
    
    //send and recieve bounds values between processors
    if(rank != 0) {
        MPI_Send(buff, 6, MPI_DOUBLE, 0, tagInput, commutator);
        MPI_Status status;
        MPI_Recv(z_buff, 2, MPI_DOUBLE, 0, tagOutput, commutator, &status);
        //printf("MPI_Recv at %d %f %f\n", rank, z_buff[0], z_buff[1]);
    }
    else {
        boundsConditions boundsZ;
        boundValue leftZbound;
        leftZbound.type = boundsNormal;
        leftZbound.k = bounds0.left.k*v_1/(1. - bounds0.left.k*u_1);
        leftZbound.r = (bounds0.left.r + bounds0.left.k*w_1)/(1. - bounds0.left.k*u_1);
        boundsZ.left = leftZbound;
        double* A = new double[size + 1];
        double* B = new double[size + 1];
        double* C = new double[size + 1];
        double* D = new double[size + 1];
        for (int i=1; i < size; i++) {
            MPI_Status status;
            MPI_Recv(buff, 6, MPI_DOUBLE, i, tagInput, commutator, &status);
            A[i] = a_0[i]*u_Nm;
            C[i] = a_0[i]*v_Nm + c_0[i] + b_0[i]*buff[0];
            B[i] = b_0[i]*buff[2];
            D[i] = d_0[i] - a_0[i]*w_Nm - b_0[i]*buff[4];
            u_Nm = buff[1];
            v_Nm = buff[3];
            w_Nm = buff[5];
        }
        boundValue rightZbound;
        rightZbound.type = boundsNormal;
        rightZbound.k = bounds0.right.k*u_Nm/(1. - bounds0.right.k*v_Nm);
        rightZbound.r = (bounds0.right.r + bounds0.right.k*w_Nm)/(1. - bounds0.right.k*v_Nm);
        boundsZ.right = rightZbound;
        
        double* z = new double[size + 1];
        TDMA(A, B, C, D, z, size + 1, boundsZ);
        for (int i=1; i < size; i++) {
            z_buff[0] = z[i];
            z_buff[1] = z[i + 1];
            //printf("send to %d %f %f\n", i, z_buff[0], z_buff[1]);
            MPI_Send(z_buff, 2, MPI_DOUBLE, i, tagOutput, commutator);
        }
        z_buff[0] = z[0];
        z_buff[1] = z[1];
        delete [] z;
        delete [] A;
        delete [] B;
        delete [] C;
        delete [] D;
    }
    for (int i = 0; i < N; i++) {
        array[i] = z_buff[0]*u[i] + z_buff[1]*v[i] + w[i];
    }
    delete [] dZero;
    delete [] u;
    delete [] v;
    delete [] w;
}

//a + c + b = d. simple method for each processor execution 
void TDMA(double* a, double* b, double* c, double* d, double* array, int N, boundsConditions bounds)
{
    if (bounds.right.type == boundsCyclic || bounds.left.type == boundsCyclic) {
        if (bounds.right.type != bounds.left.type) {
            printf("Cyclic must be both bounds");
        }
        //TODO
        return ;//TDMACyclic(a, b, c, d, out);
    }
    
    double* alp = new double[N];
    double* beta = new double[N];
    
    alp[1] = bounds.left.k;
    beta[1] = bounds.left.r;
    
    for(int i = 1; i < N - 1; ++i) {
        double ratio = (c[i] + a[i]*alp[i]);
        alp[i+1] = -b[i]/ratio;
        beta[i+1]= (d[i]-a[i]*beta[i])/ratio;
    }
    
    array[N-1] = (bounds.right.k*beta[N-1] + bounds.right.r)/(1 - bounds.right.k*alp[N-1]);
    for(int i=N-1; i>0; --i) {
        array[i-1] = alp[i]*array[i] + beta[i];
    }
    
    delete [] alp;
    delete [] beta;
}
