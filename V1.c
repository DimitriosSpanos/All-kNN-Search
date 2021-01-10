#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "auxiliary.c"
#include "auxiliary.h"
#include <mpi.h>

int *size_of_array; //sizes of the block of each process


// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


double *compute_D(double *X, double *Y,int n,int m, int d);
knnresult update_nearest(knnresult result, double * D, int n,int m, int k, int *first_call, int index_offset, int block_offset);

//! Compute distributed all-kNN of points in X
/*!

  \param  X      Data points                     [n-by-d]
  \param  n      Number of data points           [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/

knnresult distrAllkNN(double * X, int n, int d, int k){


    struct knnresult result;
    result.nidx = (int*) malloc(n*k *sizeof(int));
    if(result.nidx == NULL)
        exit(1);
    result.ndist = (double*) malloc(n*k *sizeof(double));
    if(result.ndist == NULL)
        exit(1);
    result.m = n;
    result.k = k;
/*
    ########################################
                    V1 start
    ########################################
*/

    int world_rank,m =n, p, n_incoming, first_call=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    double *D,*Y,*Z, *send;
    Y = (double *) malloc(m*d*sizeof(double));
    if(Y == NULL)
        exit(1);
    for(int i=0; i<(m*d); i++)
        Y[i] = X[i];
    MPI_Status status[2];
    MPI_Request reqsend, reqrecv;
    int offset[p];
    for(int i=0; i<p; i++){
        offset[i] = 0;
        for(int j=0; j<i; j++)
            offset[i] += size_of_array[j] / d;
    }
    for(int ring_iteration=0; ring_iteration<p; ring_iteration++){

        if(ring_iteration != p-1){
            // receive
            if (world_rank == 0){ // the first process will receive from the last one
                n_incoming = size_of_array[p-1] / d;
                Z = (double *) malloc(n_incoming*d*sizeof(double));
                if(Z == NULL)
                    exit(1);
                MPI_Irecv(Z, n_incoming*d, MPI_DOUBLE, p-1, MPI_ANY_TAG, MPI_COMM_WORLD, &reqrecv);
            }
            else{ // receive from the previous process
                n_incoming = size_of_array[world_rank-1] / d;
                Z = (double *) malloc(n_incoming*d*sizeof(double));
                if(Z == NULL)
                    exit(1);
                MPI_Irecv(Z, n_incoming*d, MPI_DOUBLE, world_rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &reqrecv);
            }
            // send
            send = (double *) malloc(m*d*sizeof(double));
            if(send == NULL)
                exit(1);
            for(int i=0; i<(m*d); i++)
                send[i] = Y[i];
            if (world_rank == p-1) // the last process will send to the first process
                MPI_Isend(send, m*d, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD, &reqsend);
            else // send to the next process
                MPI_Isend(send, m*d, MPI_DOUBLE, world_rank+1, world_rank, MPI_COMM_WORLD, &reqsend);
        }

        //hidden computation

        int chunk = 1024;
        int blocks = m / chunk;
        if(blocks == 0){
            blocks = 1;
            chunk = m;

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

        int *block_offset = (int *)malloc(blocks*sizeof(int));
        for(int i=0; i<blocks; i++){
            block_offset[i] = 0;
            for(int j=0; j<i; j++)
                block_offset[i] += block_size[j];
        }
        for(int block=0; block<blocks; block++){
            int blocked_m = block_size[block] / d;
            int end,start;
            start = block_offset[block];
            end = start + blocked_m*d ;
            double *Y_blocked = (double *) malloc( blocked_m*d* sizeof(double));
            if(Y_blocked == NULL)
                exit(1);
            for(int i=start; i<end; i++)
                Y_blocked[i-start] = Y[i];
            double *D = (double *) malloc(blocked_m *n *sizeof(double));
            if(D==NULL)
                exit(1);
            D = compute_D(X,Y_blocked,n,blocked_m,d);
            free(Y_blocked);
            result = update_nearest(result,D,n,blocked_m,k,&first_call,offset[world_rank],block_offset[block]);
            free(D);
        }
        free(block_size);free(block_offset);


        if(ring_iteration != p-1){
            MPI_Wait(&reqrecv,&status[0]);
            MPI_Wait(&reqsend,&status[1]);
            free(send);free(Y);
            m = n_incoming;
            Y = (double *) malloc(m*d*sizeof(double));
            if(Y == NULL)
                exit(1);
            for(int i=0; i<(m*d); i++)
                Y[i] = Z[i];
            free(Z);

            //update the size_of_array and offset matrices
            int * size_temp = (int *)malloc(p*sizeof(int));
            if(size_temp == NULL)
                exit(1);
            int * offset_temp = (int *)malloc(p*sizeof(int));
            if(offset_temp == NULL)
                exit(1);
            for(int i=0; i<p; i++){
                size_temp[i] = size_of_array[i];
                offset_temp[i] = offset[i];
            }
            for(int i=0; i<p; i++){
                if(i==0){
                    size_of_array[i] =size_temp[p-1];
                    offset[i] = offset_temp[p-1];
                }
                else{
                    size_of_array[i] = size_temp[i-1];
                    offset[i] = offset_temp[i-1];
                }
            }
            free(size_temp);
            free(offset_temp);
        }

    }
    free(Y);

/*
    ########################################
                     V1 end
    ########################################
*/


    return result;

}
int main()
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the number of processes
    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    double *corpus_set,*X_blocked;
    int n,d,N1;
    size_of_array = (int *)malloc(p * sizeof(int));
    if(size_of_array == NULL)
        exit(1);
    int *displacement = (int *)malloc(p * sizeof(int));
    if(displacement == NULL)
        exit(1);
    if(world_rank == 0){
        corpus_set = read_txt(&N1,&d);
        n = N1;
        int chunk = n / p;
        int remainder = n - chunk * p;
        for(int i=0; i<p; i++)
            size_of_array[i] = chunk*d;

        while(remainder != 0){
            for(int i=0; i<p; i++){
                size_of_array[i] += d;
                remainder -= 1;
                if(remainder == 0)
                    break;
            }
        }
        displacement[0] = 0;
        for(int i=1; i<p; i++){
            displacement[i] = displacement[i-1] + size_of_array[i-1];
        }
    }

    MPI_Bcast( &d, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( size_of_array, p, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( displacement, p, MPI_INT, 0, MPI_COMM_WORLD );
    n = size_of_array[world_rank] / d;
    X_blocked = (double *) malloc(size_of_array[0] *sizeof(double)); // max size of the block is the first of size_of_array
    if(X_blocked == NULL)
        exit(1);
    MPI_Scatterv( corpus_set, size_of_array, displacement, MPI_DOUBLE, X_blocked, size_of_array[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Number of neighbors: k
    int k=5;
    struct knnresult V1result;
    struct timespec tic;
    if(world_rank ==0)
        clock_gettime( CLOCK_MONOTONIC, &tic);

    V1result = distrAllkNN(X_blocked,n,d,k);
    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec toc;
    if(world_rank ==0){
        clock_gettime( CLOCK_MONOTONIC, &toc);
        printf("\n   ******************************\n     V1_duration = %f sec\n   ******************************\n\t k = %d\n",time_spent(tic, toc), k);
    }

/*
    ################################################################
       merge the small knnresults into a big one with size N1 by k
    ################################################################
*/

    struct knnresult final_result;
    if(world_rank == 0){
        final_result.nidx = (int*) malloc(N1*k *sizeof(int));
        if(final_result.nidx == NULL)
            exit(1);
        final_result.ndist = (double*) malloc(N1*k *sizeof(double));
        if(final_result.ndist == NULL)
            exit(1);
        final_result.m = N1;
        final_result.k = k;
        for(int i=0; i<n*k; i++){
            final_result.nidx[i] = V1result.nidx[i];
            final_result.ndist[i] = V1result.ndist[i];
        }
    }
    struct knnresult all_results[p-1];

    for(int pid=1; pid<p; pid++){
        if(world_rank==pid)
            MPI_Send(&n, 1, MPI_INT, 0, pid, MPI_COMM_WORLD);
        else if(world_rank == 0){
            MPI_Recv(&all_results[pid-1].m, 1, MPI_INT, pid, pid, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            all_results[pid-1].nidx = (int*) malloc(all_results[pid-1].m*k *sizeof(int));
            if(all_results[pid-1].nidx == NULL)
                exit(1);
            all_results[pid-1].ndist = (double*) malloc(all_results[pid-1].m*k *sizeof(double));
            if(all_results[pid-1].ndist == NULL)
                exit(1);
        }
        if(world_rank==pid)
            MPI_Send(V1result.ndist, n*k, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD);
        else if(world_rank == 0)
            MPI_Recv(all_results[pid-1].ndist, all_results[pid-1].m*k, MPI_DOUBLE, pid, pid, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(world_rank==pid)
            MPI_Send(V1result.nidx, n*k, MPI_INT, 0, pid, MPI_COMM_WORLD);
        else if(world_rank == 0)
            MPI_Recv(all_results[pid-1].nidx, all_results[pid-1].m*k, MPI_INT, pid, pid, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }


    if(world_rank ==0){
        int offset = n*k;
        for(int pid=1; pid<p; pid++){
            for(int i=0; i<all_results[pid-1].m*k; i++){
                final_result.ndist[offset+i] = all_results[pid-1].ndist[i];
                final_result.nidx[offset+i] = all_results[pid-1].nidx[i];
            }
            free(all_results[pid-1].ndist);
            free(all_results[pid-1].nidx);
            offset += all_results[pid-1].m *k;
        }
    }


    //! ------- PRINTING THE KNN -------
    /*
    if(world_rank == 0){
        for(int i=0; i<N1/1000; i++){
            for(int j=0; j<k; j++){
                printf("%f  ", final_result.ndist[i*k+j]);
            }
            printf("                  ");
            for(int j=0; j<k; j++){
                printf("%d  ", final_result.nidx[i*k+j]);
            }
            printf("\n");
        }
    }
    */

    // Finalize the MPI environment.
    MPI_Finalize();
    free(V1result.ndist);
    free(V1result.nidx);
    free(displacement);
    free(size_of_array);
    free(X_blocked);
    if(world_rank==0){
        free(corpus_set);
        free(final_result.ndist);
        free(final_result.nidx);
    }
    return 0;
}


double * compute_D(double *X, double *Y,int n,int m, int d){
   //nxm    nxd  dxm   nxm     nxd   dxm
    // D = (X @ X)*e1 - 2XY^T + e2 *(Y @ Y)^T
    // D =      A     -   B   +      C

    double *X_hadamarded = (double *)malloc(n*d*sizeof(double));
    for (int i=0; i<n*d; i++)
        X_hadamarded[i] = X[i]*X[i];

    double *D = (double *) malloc(n*m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, m, d, -2.0, X, d, Y, d, 0.0, D, m); // D = - B for now

    double *e1 =(double *) malloc(d*m * sizeof(double));
    double *e2 =(double *) malloc(d*n * sizeof(double));
    for (int i=0; i<(d*m); i++)
        e1[i] = 1.0;
    for (int i=0; i<(n*d); i++)
        e2[i] = 1.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, m, d, 1.0, X_hadamarded, d, e1, d, 1.0, D, m); // D = A - B for now
    free(X_hadamarded);free(e1);

    double *Y_hadamarded = (double *)malloc(m*d*sizeof(double));
    for (int i=0; i<m*d; i++)
        Y_hadamarded[i] = Y[i]*Y[i];

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        n, m, d, 1.0, e2,d, Y_hadamarded, d, 1.0, D, m); // D = A - B + C

    free(e2);free(Y_hadamarded);
    for(int i=0; i<(n*m); i++)
        D[i]= sqrt(fabs(D[i]));  // fabs because cblas gives "-0.00000" sometimes

    return D;
}

knnresult update_nearest(knnresult result, double * D, int n,int m, int k, int *first_call, int index_offset, int block_offset){
    double *min_distances = (double *) malloc(k * sizeof(double));
    if(min_distances == NULL)
        exit(1);
    int *min_dist_index = (int *) malloc(k * sizeof(int));
    if(min_dist_index == NULL)
        exit(1);
    for(int i=0; i<n; i++){
        for(int it=0; it<k; it++){
            min_distances[it] = INFINITY;
            min_dist_index[it] = -1;
        }

        for(int j=i*m; j<(i*m + m); j++){
            if(D[j] <= min_distances[k-1]){
                min_distances[k-1] = D[j];
                min_dist_index[k-1] = j - i*m + index_offset + block_offset;
                if(min_distances[k-1] < min_distances[k-2])
                    mergeSort_oneMirror(min_distances,min_dist_index,0,k-1);
            }
        }
        if(*first_call){ // if it's the first call of "update nearest" just compare them with INFINITY
            for(int it=0; it<k; it++){
                result.nidx[it+k*i] = min_dist_index[it];
                result.ndist[it+k*i] = min_distances[it];
            }
        }
        //compare with the previous nearest neighbours
        else{
            for(int it=k; it>0; it--){
                if (min_distances[k - it] < result.ndist[k*i + it - 1]){
                    result.ndist[k*i + it - 1] = min_distances[k - it];
                    result.nidx[k*i + it - 1] = min_dist_index[k - it];
                }
                else
                    break;
            }
            mergeSort_oneMirror(result.ndist,result.nidx,k*i,k*i + (k-1));
        }
    }
    if(*first_call)
        *first_call = 0;
    free(min_distances);
    free(min_dist_index);
    return result;
}
