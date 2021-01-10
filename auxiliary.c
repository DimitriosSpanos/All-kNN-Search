#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


void SWAP(double *x, double *y) {
    double temp = *x;
    *x = *y;
    *y = temp;
}

double *read_txt(int *N1, int *d){
    char data_points[] = "ColorMoments.txt";
    double *corpus_set;
    FILE *f = fopen(data_points,"r");
    int counter;
    counter = fscanf(f, "%d %d\n",N1, d);
    //!uncomment this to make a test with N = 7000, d = ... locally
    //*N1 = 7000;
    corpus_set = (double *)malloc(*N1**d*sizeof(double));
    for(int i=0; i<*N1; i++){
        for(int j=0; j<*d; j++){
            if (j != *d-1)
                counter = fscanf(f, "%lf\t",&corpus_set[i**d + j]);
            else
                counter = fscanf(f, "%lf\n",&corpus_set[i**d + j]);
        }
    }
    fclose(f);
    return corpus_set;
}





void merge_oneMirror(double *arr,int *mirror, int l, int m, int r){
    int k;
    int n1 = m - l + 1;
    int n2 = r - m;

    double *L,*R;
    int *L_mirror,*R_mirror;
    L=(double *) malloc(n1 * sizeof(double));
    R =(double *) malloc(n2 * sizeof(double));
    L_mirror=(int *) malloc(n1 * sizeof(int));
    R_mirror =(int *) malloc(n2 * sizeof(int));


    for (int i = 0; i < n1; i++){
        L[i] = arr[l + i];
        L_mirror[i] = mirror[l+i];
    }

    for (int j = 0; j < n2; j++){
        R[j] = arr[m + 1 + j];
        R_mirror[j] = mirror[m+1+j];
    }

    int i = 0;
    int j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            mirror[k] = L_mirror[i];
            i++;
        }
        else {
            arr[k] = R[j];
            mirror[k] = R_mirror[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        mirror[k] = L_mirror[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        mirror[k] = R_mirror[j];
        j++;
        k++;
    }
}


void mergeSort_oneMirror(double *arr,int *mirror, int l, int r){
    if (l < r) {
        int m = l + (r - l) / 2;
        #pragma omp task
        mergeSort_oneMirror(arr,mirror, l, m);
        mergeSort_oneMirror(arr,mirror, m + 1, r);
        #pragma omp taskwait
        merge_oneMirror(arr,mirror, l, m, r);
    }
}




double time_spent(struct timespec start,struct timespec end_){
        struct timespec temp;
        if ((end_.tv_nsec - start.tv_nsec) < 0)
        {
                temp.tv_sec = end_.tv_sec - start.tv_sec - 1;
                temp.tv_nsec = 1000000000 + end_.tv_nsec - start.tv_nsec;
        }
        else
        {
                temp.tv_sec = end_.tv_sec - start.tv_sec;
                temp.tv_nsec = end_.tv_nsec - start.tv_nsec;
        }
        return (double)temp.tv_sec +(double)((double)temp.tv_nsec/(double)1000000000);

}



double partition_of_quick(double *a, int left, int right, int pivotIndex){

	double pivot = a[pivotIndex];
	SWAP(&a[pivotIndex], &a[right]);


	int pIndex = left;
	int i;

	for (i = left; i < right; i++){
		if (a[i] <= pivot){
			SWAP(&a[i], &a[pIndex]);
			pIndex++;
		}
	}

	SWAP(&a[pIndex], &a[right]);

	return pIndex;
}


double quickselect(double *A, int left, int right, int k){

	if (left == right)
		return A[left];
	int pivotIndex = (left + right)/2;
	pivotIndex = partition_of_quick(A, left, right, pivotIndex);


	if (k == pivotIndex)
		return A[k];
	else if (k < pivotIndex)
		return quickselect(A, left, pivotIndex - 1, k);
	else
		return quickselect(A, pivotIndex + 1, right, k);
}



