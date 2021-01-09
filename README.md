# Parallel-Exercise1

To test locally, you can uncomment the assignment on *N1 in the **read_txt** function
and have a smaller corpus. Furthermore, you can uncomment the printing of the result in the **main** function.

## **V0**
The files that pertain to the V0 part of the exercise are
**V0.c**, **auxiliary.c** and **auxiliary.h**. Choose the appropriate 
txt by giving it to the **read_txt** function in the **auxiliary.c**, as follows.

```
char data_points[] = "corpus.txt";
```
The txt must have the following format to fuction properly.
There are n=3 lines, where each line represents one point.
Each line has d=5 floats (coordinates) that are delimited by \t.

```
3 5
2.59E+00  4.69E-01	2.07E+01	3.23E-01	9.68E-03
0.00E+00	9.05E-02	1.77E-01	4.58E-01	7.18E-02	
3.86E+00	6.46E-01	1.81E+01	2.34E-01	3.07E-02	
```
## **V1**
The files that pertain to the V1 of the exercise are **V1.c**, **auxiliary.c** and **auxiliary.h**.

The **Makefile** can be used to make a test locally using the following command. (p = number of processors, e.g 4)

```
make testV1 p=4
```
Alternatively, in the **HPC** the following command can be used.
```
srun -n 4 ./V1
```
## **V2**
The files that pertain to the V2 of the exercise are **V2.c**, **auxiliary.c**, **auxiliary.h**, **VPT.c** and **VPT.h**.

The **Makefile** can be used to make a test locally using the following command. (p = number of processors, e.g 4)

```
make testV2 p=4
```
Alternatively, in the **HPC** the following command can be used.
```
srun -n 4 ./V2
```

Repo for the second exercise of course 050 - Parallel and Distributed Systems, Aristotle University of Thessaloniki, Dpt. of Electrical & Computer Engineering.
