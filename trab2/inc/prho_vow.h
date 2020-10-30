#ifndef PRHO_VOW_H_INCLUDED
#define PRHO_VOW_H_INCLUDED

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <sys/param.h>    // Attempt to define endianness
#include <endian.h>      // Attempt to define endianness


#define NUM_IT_FUNCTION_INTERVALS   32
#define NO_KEY_FOUND                -1

#define VOW     1
#define TOHA    2
#define HARES   3

#define ESP     ' '
#define LF      10
#define DEL     127
#define DEBUG   0
#define SHOW    1

#define INF                         0x7FFFFFFFFFFFFFFF
#define MAX_POINTS                  1000


#define EMPTY_SLOT_NOT_FOUND        -1    // unable to find empty slot
#define STORED                       0
#define GOOD_COLLISION               1
#define UNFRUITFUL_COLLISION         2


// Hash table size
#define TABLESIZE                  hashsize(20)


// Macros for bit values
#define hashsize(n) ((uint32_t)1<<(n))
#define hashmask(n) (hashsize(n)-1)
#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))


// TYPE DEFINITIONS

typedef struct type_point {
   long long int x;
   long long int y;
   long long int r;
   long long int s;
} POINT_T;



typedef struct PointSet {
   POINT_T itpoint[NUM_IT_FUNCTION_INTERVALS];
} IT_POINT_SET;


typedef  unsigned int uint32_t;



// FUNCTION PROTOTYPES

long long calc_y2(long long x, long long a, long long b, long long mod);
int perfsqr(POINT_T P, long long a, long long b, long long mod);
long long find_points (long long a, long long b, long long p, POINT_T * G, int show);
long long in_curve(POINT_T P, long long a, long long b, long long p);
long long calc_order(POINT_T Q, long long a, long long b, long long mod);

long long multiplicative_inverse(long long num, long long modulo);
POINT_T add_2P(POINT_T p, long long a, long long mod);
POINT_T add_PQ(POINT_T p, POINT_T q, long long mod);
POINT_T addpoints(POINT_T P, POINT_T Q, long long a, long long p, long long order);
POINT_T subpoints(POINT_T p, POINT_T q, long long a, long long mod);
POINT_T multpoint(long long n, POINT_T P, long long a, long long mod, long long order);
long long equal(POINT_T X1, POINT_T X2);
void printP(POINT_T X);
long long calc_order(POINT_T G, long long a, long long b, long long mod);
void calc_P_sums(POINT_T P, long long a, long long p, long long order, int nbits);
void calc_Q_sums(POINT_T Q, long long a, long long p, long long order, int nbits);
void calc_Q(POINT_T *pQ, POINT_T *Psums, long long k, int nbits, long long a, long long p);
int store_point(POINT_T X);
long long get_rand(long long order);

void setup(POINT_T *X, POINT_T P, POINT_T Q, long long a, long long p, long long order,
             const int L, const int nbits, const long long nworkers, int alg);
POINT_T nextpoint(POINT_T P, long long a, long long p, long long order, int L, POINT_T *Triplet);
long long int get_k(long long c1, long long d1, long long c2, long long d2,
                    long long order);
void calc_tripletset(POINT_T *tripletset, POINT_T P, POINT_T Q, long long a,
                   long long p, long long order, int L, int nbits);
void rand_itpset(IT_POINT_SET *itpset, POINT_T *Psums, POINT_T *Qsums,
            long long a, long long p, long long order, int L, int nbits,
            int algorithm, int tid);
void check_slot_and_store(POINT_T X, POINT_T *Y, int token, int *retval);
long long int calc_k(POINT_T P, POINT_T Q, long long a, long long p, long long order,
                    long long *numits, int L, const long long nworkers, int alg);
void initial_point(POINT_T *init_point, POINT_T *Psums, POINT_T *Qsums, int nbits,
                   int gtid, int alg, long long nworkers, long long a,
                   long long p, long long order);

void calc_iteration_point_set(POINT_T **itpset_P, POINT_T *itsetbase, POINT_T *Psums, POINT_T *Qsums, int tid,
                              long long a, long long p, long long order, int L, int nbits, int algorithm);

void calc_iteration_point_set2(IT_POINT_SET *itpset, IT_POINT_SET *itsetbase, POINT_T *Psums, POINT_T *Qsums, int tid,
                              long long a, long long p, long long order, int L, int nbits, int algorithm);


// Timing functions
double get_wall_time();
double get_cpu_time();

// Hash functions
uint32_t hashword (const uint32_t *k, size_t length, uint32_t initval);
void     hashword2(const uint32_t *k, size_t length, uint32_t *pinit1, uint32_t *pinit2);

// Free memory functions
void free_allocated_memory();


#endif // PRHO_VOW_H_INCLUDED
