#ifndef VPT_H_INCLUDED
#define VPT_H_INCLUDED

// type definition of vptree
typedef struct Vptree
{
    //keep vantage point as an array
    double *vp;
    //median distance of vp to other points
    double md ;
    //vantage point index in the original set
    int idx;
    //vantage point subtrees
    struct Vptree *inner;
    struct Vptree *outer;

}vptree;

vptree * createVPT(double *X, int n, int d, int offset);
vptree * getInner(vptree * T);
vptree * getOuter(vptree * T);
double getMD(vptree * T);
double * getVP(vptree * T);
int getIDX(vptree * T);
void delete_tree(vptree *T);


#endif // VPT_H_INCLUDED
