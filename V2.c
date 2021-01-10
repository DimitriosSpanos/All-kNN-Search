#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "auxiliary.c"
#include "auxiliary.h"
#include <mpi.h>
#include "VPT.h"
#include "VPT.c"

int *size_of_array; //sizes of the block of each process
// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

void searchVPT(vptree *T, double *point, double *point_DIST, int *point_IDX, int d, int k);
void update_nearest(double * vp, int vpIDX, double point[], double point_DIST[], int point_IDX[], double * distance, int d, int k);
int checkIntersection(double median, double distance, double max_distance);

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
    for(int i=0; i<(n*k); i++){
        result.ndist[i] = INFINITY;
        result.nidx[i] = -1;
    }

/*
    ########################################
                    V2 start
    ########################################
*/

    int world_rank,m =n, p, n_incoming;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int offset[p]; // this offset helps with indices of the queries
    for(int i=0; i<p; i++){
        offset[i] = 0;
        for(int j=0; j<i; j++)
            offset[i] += size_of_array[j] / d;
    }

    double *Y,*Z, *send;
    Y = (double *) malloc(m*d*sizeof(double));
    for(int i=0; i<(m*d); i++)
        Y[i] = X[i];

    MPI_Status status[2];
    MPI_Request req[2];
    for(int ring_iteration=0; ring_iteration<p; ring_iteration++){

        if(ring_iteration != p-1){

            // receive
            int destination, source;
            if (world_rank == 0)
                source = p-1;
            else
                source = world_rank-1;
            n_incoming = size_of_array[source] / d;
            Z = (double *) malloc(n_incoming*d*sizeof(double));
            MPI_Irecv(Z, n_incoming*d, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &req[0]);

            // send
            send = (double *) malloc(m*d*sizeof(double));
            for(int i=0; i<(m*d); i++)
                send[i] = Y[i];
            if (world_rank == p-1)
                destination = 0;
            else
                destination = world_rank+1;
            MPI_Isend(send, m*d, MPI_DOUBLE, destination, 0, MPI_COMM_WORLD, &req[1]);
        }


        //hidden computation
        vptree *T = createVPT(Y, m, d, offset[world_rank]);
        for(int i=0; i<n; i++){
            double point[d];
            double point_DIST[k];
            int point_IDX[k];
            for(int j=0; j<d; j++)
                point[j] = X[i*d+j];
            for(int j=0; j<k; j++){
                point_DIST[j] = result.ndist[i*k + j];
                point_IDX[j] = result.nidx[i*k + j];
            }

            searchVPT(T, point, point_DIST, point_IDX, d, k);

            for(int j=0; j<k; j++){
                result.ndist[i*k + j] = point_DIST[j];
                result.nidx[i*k + j] = point_IDX[j];
            }
        }
        delete_tree(T);


        if(ring_iteration != p-1){
            MPI_Waitall(2,req,status);
            free(send);free(Y);
            m = n_incoming;
            Y = (double *) malloc(m*d*sizeof(double));
            for(int i=0; i<(m*d); i++)
                Y[i] = Z[i];
            free(Z);

            //update the size_of_array and offset matrices
            int * size_temp = (int *)malloc(p*sizeof(int));
            int * offset_temp = (int *)malloc(p*sizeof(int));
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
                     V2 end
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
    int *displacement = (int *)malloc(p * sizeof(int));
    if(size_of_array == NULL)
        exit(1);

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
    MPI_Scatterv( corpus_set, size_of_array, displacement, MPI_DOUBLE, X_blocked, size_of_array[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Number of neighbors: k
    int k=5;
    struct knnresult V2result;
    struct timespec tic;
    if(world_rank ==0)
        clock_gettime( CLOCK_MONOTONIC, &tic);

    V2result = distrAllkNN(X_blocked,n,d,k);
    MPI_Barrier(MPI_COMM_WORLD);

    struct timespec toc;
    if(world_rank ==0){
        clock_gettime( CLOCK_MONOTONIC, &toc);
        printf("\n   ******************************\n     V2_duration = %f sec\n   ******************************\n\t k = %d\n",time_spent(tic, toc), k);
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
            final_result.nidx[i] = V2result.nidx[i];
            final_result.ndist[i] = V2result.ndist[i];
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
            MPI_Send(V2result.ndist, n*k, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD);
        else if(world_rank == 0)
            MPI_Recv(all_results[pid-1].ndist, all_results[pid-1].m*k, MPI_DOUBLE, pid, pid, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(world_rank==pid)
            MPI_Send(V2result.nidx, n*k, MPI_INT, 0, pid, MPI_COMM_WORLD);
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

    MPI_Finalize();
    free(V2result.ndist);
    free(V2result.nidx);
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


void searchVPT(vptree *T, double point[], double point_DIST[], int point_IDX[], int d , int k){

    if(T == NULL)
        return;

    double distance = 0;
    update_nearest(getVP(T), getIDX(T), point, point_DIST, point_IDX, &distance, d, k);
    double max_dist =point_DIST[k-1];

    if (getInner(T) == NULL && getOuter(T) == NULL)
        return;

    if(distance < getMD(T)){ // give priority to Inner
        if(distance - max_dist <= getMD(T))
            searchVPT(getInner(T), point, point_DIST, point_IDX, d, k);

        if(checkIntersection(getMD(T), distance, max_dist))
            searchVPT(getOuter(T), point, point_DIST, point_IDX, d, k);
    }
    else{       // give priority to Outer
        if(checkIntersection(getMD(T), distance, max_dist))
            searchVPT(getOuter(T), point, point_DIST, point_IDX, d, k);

        if(distance - max_dist <= getMD(T))
            searchVPT(getInner(T), point, point_DIST, point_IDX, d, k);
    }
}


void update_nearest(double * vp, int vpIDX, double point[], double point_DIST[], int point_IDX[], double * distance, int d, int k){

    for (int i=0; i<d; i++){
        double coord_diff = vp[i]-point[i];
        *distance += coord_diff*coord_diff;
    }

    *distance= sqrt(fabs(*distance));

    if (*distance < point_DIST[k-1]){
        point_DIST[k-1] = *distance;
        point_IDX[k-1] = vpIDX;
        if(point_DIST[k-1] < point_DIST[k-2])
            mergeSort_oneMirror(point_DIST,point_IDX,0,k-1);
    }
}


int checkIntersection(double median, double distance, double max_distance){
    return (distance + max_distance >=   median);
}
