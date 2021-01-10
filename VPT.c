#include <stdio.h>
#include <stdlib.h>
#include "VPT.h"
#include <math.h>
#include <omp.h>
#include <mpi.h>

double * calcDistance(double * X ,int n, int d){

    double *distances = (double *)calloc(n,sizeof(double));
    if (distances == NULL)
        exit(1);

    for(int i=0; i<n-1; i++){
        for (int j=0; j<d; j++){
            double coord_diff = X[i*d + j]-X[(n-1)*d + j];
            distances[i] += coord_diff*coord_diff;
        }
        distances[i] = sqrt(fabs(distances[i]));
    }
    return distances;
}

vptree * buildVPT_recursively(double *X, int n, int d, int *xID){
    double * dist,*distances=(double *)malloc(n*sizeof(double)) ;
    vptree *T = NULL;
    int innerSize= 0, outerSize = 0;
    //allocate 0 space and realloc later, as innerSize is not predefined
    double *innerX = (double *)malloc(innerSize*d * sizeof(double));
    int *innerID = (int *)malloc(innerSize * sizeof(int));
    double *outerX = (double *)malloc(outerSize*d * sizeof(double));
    int *outerID = (int *)malloc(outerSize * sizeof(int));

    if(n==0){
        free(distances);
        free(dist);
        free(innerX);
        free(outerX);
        free(innerID);
        free(outerID);
        return T;
    }

    T= (vptree *)calloc(1, sizeof (vptree));
    T->vp = (double *)malloc(d * sizeof(double));
    T->idx = xID[n-1]; // last element of the X array
    for (int i=0; i<d; i++)
        T->vp[i] = X[(n-1)*d + i];

    if(n==1){
        free(distances);
        free(dist);
        free(innerX);
        free(outerX);
        free(innerID);
        free(outerID);
        return T;
    }

    dist= calcDistance(X, n, d);
    for(int i=0; i<n; i++)
        distances[i]=dist[i];

    if ((n-1)%2 != 0) // find median if odd elements
        T->md = quickselect(dist, 0, n-1, n/2);
    else              // find median if even elements
        T->md = ( quickselect(dist, 0, n-1, n/2+1) + quickselect(dist,0,  n-1, n/2) ) / 2;


    for(int i=0 ; i<n-1 ; i++){
        if(distances[i]<= T->md){ // everything with less than median goes Inner
            innerSize++;
            innerX = realloc(innerX , innerSize*d * sizeof(double));
            innerID = realloc(innerID , innerSize * sizeof(int));
            innerX[innerSize*d - 1] = i;
            innerID[innerSize-1] = xID[i];
        }
        else{                   // everything with more than median goes Outer
            outerSize++;
            outerX = realloc(outerX, outerSize*d * sizeof(double));
            outerID = realloc(outerID , outerSize * sizeof(int));
            outerX[outerSize*d - 1] = i;
            outerID[outerSize-1] = xID[i];
        }
    }

    for(int i=0 ; i<innerSize; i++){
        for(int j=0; j<d; j++)
            innerX[i*d+j]=  X[(int)(innerX[i*d + d -1])*d+j];
    }

    for(int i=0 ; i<outerSize; i++){
        for(int j=0; j<d; j++)
            outerX[i*d+j]=X[(int)(outerX[i*d + d - 1])*d+j];
    }


    #pragma omp task
    T->inner = buildVPT_recursively(innerX, innerSize, d,innerID);
    T->outer = buildVPT_recursively(outerX, outerSize, d, outerID);
    #pragma omp taskwait
    free(innerX);free(outerX);free(innerID);free(outerID);
    free(distances);free(dist);
    return T;
}

vptree * createVPT(double *X, int n, int d, int offset){

    int *Xid = (int *)malloc(n*sizeof(int));

    for(int i=0;i<n;i++)
        Xid[i] = i + offset;

    vptree *T=buildVPT_recursively(X, n, d, Xid);
    free(Xid);
    return T;
}


vptree * getInner(vptree * T){
    return T->inner;
}

vptree * getOuter(vptree * T){
    return T->outer;
}

double getMD(vptree * T){
    return T->md;
}

double *getVP(vptree *T){
    return T->vp;
}

int getIDX(vptree *T){
    return T->idx;
}

void delete_tree(vptree *T){
    if(T==NULL)
        return;
    delete_tree(T->inner);
    delete_tree(T->outer);
    free(T->vp);
    free(T);
}
