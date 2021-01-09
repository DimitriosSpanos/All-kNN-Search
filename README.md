# Parallel-Exercise2

**mmio.c** and **mmio.h** are used to extract the info from the .mtx files.
The source code is given in the following link that you provided.
https://math.nist.gov/MatrixMarket/mmio-c.html

**validity.py** was used for the 3.1.1 section of this exercise.
**plot.py** was used to create the diagrams for the report.

## **2 Preliminaries**
The files that pertain to the preliminaries part of the exercise are
**main.c**, **main_openmp.c** and **main_cilk.c**. Choose the appropriate 
matrix by giving it as an argument to main(), as follows.

```
./main matrix_name.mtx
./main_openmp matrix_name.mtx
./main_cilk matrix_name.mtx
```
Alternatively, one can use the **Makefile** to make a test that runs all of the
above programs, like so.
```
make testV3 matrix="matrix_name.mtx"
```
## **3 Main part**
The files that pertain to the main part of the exercise are **make_graph.c**
**main3.c**, **main3_openmp.c**, **main3_cilk.c** and **main3_thread.c**.

It's vital that the **make_graph.c** file is run first so that a *graph.mtx*
file is created. The remaining programs use that as their input.
The **make_graph.c** works like the previous ones, as follows.
```
./make_graph matrix.mtx
```
The **Makefile**  can be used to make a test that runs all of the
above programs, like so.
```
make testV4
```


Repo for the first exercise of course 050 - Parallel and Distributed Systems, Aristotle University of Thessaloniki, Dpt. of Electrical & Computer Engineering.
