#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "auxiliary.c"
#include "auxiliary.h"

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


knnresult kNN(double * X, double * Y, int n, int m, int d, int k);




//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  \param  X      Corpus data points              [n-by-d]
  \param  Y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult kNN(double * X, double * Y, int n, int m, int d, int k){




    struct knnresult result;
    result.nidx = (int*) malloc(m*k *sizeof(int));
    if(result.nidx == NULL)
        exit(1);
    result.ndist = (double*) malloc(m*k *sizeof(double));
    if(result.ndist == NULL)
        exit(1);
    result.m = m;
    result.k = k;

/*
    ########################################
                    V0 start
    ########################################
*/


    int chunk = 1024;
    int blocks = m / chunk;
    if(blocks == 0){
        chunk = m;
        blocks = 1;
    }
    int *block_size = (int *)malloc(blocks*sizeof(int));
    int remainder = m - chunk * blocks;
    for(int i=0; i<blocks; i++)
        block_size[i] = chunk*d;
    while(remainder != 0){
        for(int i=0; i<blocks; i++){
            block_size[i] += d;
            remainder -= 1;
            if(remainder == 0)
                break;
        }
    }

    int offset[blocks]; // this handles the correction of indices (because of blocking)
    for(int i=0; i<blocks; i++){
        offset[i] = 0;
        for(int j=0; j<i; j++)
            offset[i] += block_size[j];
    }



    //nxm    nxd  dxm   nxm     nxd   dxm
    // D = (X @ X)*e1 - 2XY^T + e2 *(Y @ Y)^T
    // D =      A     -   B   +      C


    double *X_hadamarded;
    X_hadamarded = (double *) malloc(n*d * sizeof(double));
    if(X_hadamarded == NULL)
        exit(1);
    for(int i=0; i<(n*d); i++)
        X_hadamarded[i] = X[i] * X[i]; // X @ X

    for(int block=0; block<blocks; block++){
        int blocked_m = block_size[block] / d;
        int end,start;
        start = offset[block];
        end = start + blocked_m*d ;

        double *Y_blocked = (double *) malloc( blocked_m*d* sizeof(double));
        if(Y_blocked == NULL)
            exit(1);
        for(int i=start; i<end; i++)
            Y_blocked[i-start] = Y[i];

        double *D = (double *) malloc(n*blocked_m * sizeof(double));
        if(D == NULL)
            exit(1);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, blocked_m, d, -2.0, X, d, Y_blocked, d, 0.0, D, blocked_m); // D = - B for now

        double *e1,*e2;
        e1 = (double *) malloc(d*blocked_m * sizeof(double));
        if(e1 == NULL)
            exit(1);
        e2 = (double *) malloc(n*d * sizeof(double));
        if(e2 == NULL)
            exit(1);
        for (int i=0; i<(d*blocked_m); i++)
            e1[i] = 1.0;
        for (int i=0; i<(n*d); i++)
            e2[i] = 1.0;


        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, blocked_m, d, 1.0, X_hadamarded, d, e1, d, 1.0, D, blocked_m); // D = A - B for now
        free(e1);

        for (int i=0; i<(blocked_m*d); i++)
            Y_blocked[i] = Y_blocked[i] * Y_blocked[i];  // Y = Y @ Y
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, blocked_m, d, 1.0, e2,d, Y_blocked, d, 1.0, D, blocked_m); // D = A - B + C
        free(Y_blocked);free(e2);

        for(int i=0; i<(n*blocked_m); i++){
            if(D[i] < 0.000001)
                D[i] = 0;
        }

    /*
         ####################################
            Find result.nidx, result.ndist
         ####################################
    */

        for(int i=start/d; i<end/d; i++){ //for each Y element
            //find k-minimum distances of col i
            double * min_distances = (double *) malloc(k * sizeof(double));  // heap that holds temporary k - minimum distances
            if(min_distances == NULL)
                exit(1);
            int * min_dist_index = (int *) malloc(k * sizeof(int));
            if(min_dist_index == NULL)
                exit(1);
            for(int it=0; it<k; it++){
                min_distances[it] = INFINITY;
                min_dist_index[it] = -1;
            }
            for(int j=0; j<n*blocked_m; j+=blocked_m){
                if(D[j+i-start/d] <= min_distances[k-1]){
                    min_distances[k-1] = D[j+i-start/d];
                    min_dist_index[k-1] = (j+i-start/d)/blocked_m;
                    if(min_distances[k-1] < min_distances[k-2]) // sort the heap only if it is required
                        mergeSort_oneMirror(min_distances,min_dist_index,0,k-1);
                }
            }
            for(int it=0; it<k; it++){
                result.nidx[it+k*i] = min_dist_index[it];
                result.ndist[it+k*i] = min_distances[it];
            }
            free(min_distances);
            free(min_dist_index);
        }
        free(D);
    }

/*
    ########################################
                     V0 end
    ########################################
*/


    free(X_hadamarded);


    for(int i=0; i<(m*k); i++)
        result.ndist[i]= sqrt(fabs(result.ndist[i]));
    return result;
}


int main()
{
    int N1,d;

    double *corpus_set = read_txt(&N1,&d);
    // Number of neighbors: k
    int k = 5;


    struct timespec tic;
    clock_gettime( CLOCK_MONOTONIC, &tic);
    struct knnresult V0result = kNN(corpus_set,corpus_set,N1,N1,d,k);


    struct timespec toc;
    clock_gettime( CLOCK_MONOTONIC, &toc);

    printf("\n   ******************************\n     V0_duration = %f sec\n   ******************************\n\t k = %d\n",time_spent(tic, toc), k);

    free(corpus_set);

    //! ------- PRINTING THE KNN -------
    /*
    for(int i=0; i<N1; i++){
        for(int j=0; j<k; j++){
            printf("%f  ", V0result.ndist[i*k+j]);
        }
        printf("                  ");
        for(int j=0; j<k; j++){
            printf("%d  ", V0result.nidx[i*k+j]);
        }
        printf("\n");
    }
    */

    free(V0result.ndist);
    free(V0result.nidx);
    return 0;
}
