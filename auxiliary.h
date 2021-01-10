#ifndef AUXILIARY_H_INCLUDED
#define AUXILIARY_H_INCLUDED


double *read_txt(int *n, int *d);
void merge_oneMirror(double *arr,int *mirror, int l, int m, int r);
void mergeSort_oneMirror(double *arr,int *mirror, int l, int r);
double time_spent(struct timespec start,struct timespec end_);
double quickselect(double A[], int left, int right, int k);
double partition_of_quick(double a[], int left, int right, int pivotIndex);

#endif // AUXILIARY_H_INCLUDED
