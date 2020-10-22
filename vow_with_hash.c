// Define CUDA function directives as spaces
#define __global__
#define __device__

#include <prho_vow.h>
#include "C:\MinGW\include\time.h"
#include <hash3_code.h>


// Defines that the user can control
#define MAX_RUNS                    30
#define NUM_WORKERS                200
#define MODULO_NUM_BITS             24


// GLOBAL VARIABLES

// CONSTANT VALUED VARIABLES
const int algorithm = VOW;
const int L = NUM_IT_FUNCTION_INTERVALS;
const int nbits = MODULO_NUM_BITS;
const int nworkers = NUM_WORKERS;

// NON-CONSTANT VALUED GLOBAL VARIABLES
long long n_unfruitfulcollisions = 0;
long long kvaluefound;

POINT_T   hashtable[TABLESIZE];
POINT_T   EMPTY_POINT = {0, 0, 0, 0};
POINT_T   *Psums, *Qsums, *Tsums;
POINT_T   X[NUM_THREADS];
POINT_T   initX[NUM_THREADS];

IT_POINT_SET itPsetBase;

IT_POINT_SET itPsets[NUM_THREADS];


////////////////////////////////////////////////////////////////////////////
//
// Utility Functions
//
////////////////////////////////////////////////////////////////////////////


void GetXY(int *x, int *y)
{
  CONSOLE_SCREEN_BUFFER_INFO  csbInfo;
  GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbInfo);
  *x = csbInfo.dwCursorPosition.X;
  *y = csbInfo.dwCursorPosition.Y;
}

int wherex() {
    int x, y;
    GetXY(&x, &y);
    return x;
}

int wherey() {
    int x, y;
    GetXY(&x, &y);
    return y;
}


double get_wall_time(){
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}

double get_cpu_time(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;

    }
}


void printP(POINT_T X)  {
    printf("(%7lld, %7lld), r = %8lld, s = %8lld", X.x, X.y, X.r, X.s);
}



////////////////////////////////////////////////////////////////////////////
//
// Functions to handle elliptic point aritmetic
//
////////////////////////////////////////////////////////////////////////////

POINT_T add_2P(POINT_T p, long long a, long long mod)
{
	POINT_T val;
	long long lambda, num, den, invden;

	if (p.y == 0)
    {
    	// 2P is equal to O(0, 0)
    	val.x = 0;
    	val.y = 0;
    	return(val);
    }

	num = (3*p.x*p.x + a) % mod;
	while (num < 0) {
		num = num + mod;
	}
	den = 2*p.y;
	while (den < 0) {
        den += mod;
    }
    den = den % mod;

	invden = multiplicative_inverse(den, mod);
	lambda = (num * invden) % mod;

	val.x = (lambda*lambda - p.x - p.x);
	while (val.x < 0) {
	    val.x += mod;
	}
	val.x = val.x % mod;

	val.y = (lambda*(p.x - val.x) - p.y);
		while (val.y < 0) {
	    val.y += mod;
	}
	val.y = val.y % mod;

	return(val);
}


POINT_T add_PQ(POINT_T p, POINT_T q, long long mod)
{
	POINT_T val;
	long long lambda, num, den, invden;

	// Test if p or q is the point at infinite
	if (p.x == 0  &&  p.y == 0)  return(q);
	if (q.x == 0  &&  q.y == 0)  return(p);

	if (p.x == q.x) {
    	val.x = 0;
    	val.y = 0;
    	return(val);
    }

	num = (q.y - p.y);
	while (num < 0) {
		num = num + mod;
	}
	num = num % mod;

	den = q.x - p.x;
	while (den < 0) {
        den += mod;
    }
    den = den % mod;

	invden = multiplicative_inverse(den, mod);
	lambda = (num*invden) % mod;

	val.x = (lambda*lambda - p.x - q.x);
	while (val.x < 0) {
	    val.x += mod;
	}
	val.x = val.x % mod;

	val.y = (lambda*(p.x - val.x) - p.y);
	while (val.y < 0) {
	    val.y += mod;
	}
	val.y = val.y % mod;

	return(val);
}


POINT_T subpoints(POINT_T p, POINT_T q, long long a, long long mod)
{
	POINT_T val;

	q.y = -q.y;
	while (q.y < 0) {
		q.y += mod;
	}

	if (p.x==q.x && p.y==q.y)  val = add_2P(p, a, mod);
	else val = add_PQ(p, q, mod);

	return(val);
}


POINT_T addpoints(POINT_T P, POINT_T Q, long long a , long long p, long long order)
{
	POINT_T val;

	if (P.x==Q.x && P.y==Q.y)  val = add_2P(P, a, p);
	else val = add_PQ(P, Q, p);

	if (order != 0) {
        val.r = (P.r + Q.r) % order;
        val.s = (P.s + Q.s) % order;
	}

	return(val);
}


POINT_T multpoint(long long n, POINT_T P, long long a, long long mod, long long order)
{
	long long i=1;
	POINT_T val = P;

	if (n == 1) return(P);
	if (n < 0)  { val.x = -1;  val.y = -1;  return(val); }
	if (n == 0) { val.x = 0;   val.y = 0;   return(val); }

	do {
		val = addpoints(val, P, a, mod, order);
		i++;
	} while (i < n);

	return(val);
}


long long multiplicative_inverse(long long num, long long modulo)
{
	long long val=0;
    long long a, b, q, r, x, y;
	long long rest, r1, r2, x1=1, x2=0, y1=0, y2=1;

	if (num == 1) return(1);

    if (num >= modulo) {   a = num;   b = modulo;   }
    else {   a = modulo;   b = num;   }

    r2 = a;
    r1 = b;

    long long i = 1, gcd;

    do {
    	rest = r2 % r1;

    	if (rest == 0) {

    		if (DEBUG) {
    		    printf("\n\n_______________________________________________________________________________\n\n\n");
    		    printf("WILL BREAK:   rest = (r2 % r1) = 0      r2=%lld    r1=%lld\n\n", r2, r1);
    		    printf("_______________________________________________________________________________\n\n");
    	    }

    	    break;
    	}

    	q = r2/r1;
    	r = rest;
    	x = x1 - q*x2;
    	y = y1 - q*y2;

    	if (DEBUG) {
            printf("\n\n\na[%lld]=%lld   b[%lld]=%lld   q[%lld]=%lld", i, r2, i, r1, i, q);
		    printf("    r[%lld]=%lld    x[%lld]=%lld    y[%lld]=%lld\n\n", i, r, i, x, i, y);
		    printf("%lld = %lld*(%lld) + %lld\n\n", r2, q, r1, r);
		    printf("%lld = %lld*(%lld) + %lld*(%lld)\n", r, a, x, b, y);
	    }

		/* At his point  r = a*x + b*y  */
    	if (r != (a*x + b*y))  {
    		printf("\nError: r != (a * x   +   b * y)\n\n");
    		printf("%lld  !=  (%lld * %lld  +  %lld * %lld)      q=%lld\n", r, a, x, b, y, q);
    		break;
        }

        r2 = r1;
		r1 = r;

		x1 = x2;
		x2 = x;

		y1 = y2;
		y2 = y;

		i++;
    } while (1);

	if (r == 0) {
	    if (num <= modulo) gcd = num;
		else gcd = modulo;
	}
	else gcd = r;

	if (DEBUG) {
	    printf("\ngcd (%lld, %lld) = %lld", num, modulo, gcd);
	    printf("      x = %lld      y = %lld\n\n", x, y);
	    printf("%lld = %lld*(%lld) + %lld*(%lld)\n\n", gcd, a, x, b, y);
    }

	if (gcd == 1)  {
		if (num >= modulo)     /*  a == num,  check x */
		    if (x > 0) {

		    	printf("The multiplicative inverse of %lld mod %lld is %lld\n\n", a, b, x);
		    	if (DEBUG) {
		            printf("The multiplicative inverse of %lld mod %lld is %lld\n\n", a, b, x);
		            printf("(%lld * %lld) mod %lld = %lld\n\n", a, x, b, a*x % b);
		        }
		        val = x;
		    }
		    else {
		    	if (DEBUG)  {
		            printf("The multiplicative inverse of %lld mod %lld is (%lld - %lld) = %lld\n\n", a, b, b, -x, b+x);
		            printf("(%lld * %lld) mod %lld = %lld\n\n", a, b+x, b, (a*(b+x)) % b);
		        }
		        val = b+x;
	        }
	    else     /* b == num,  check y */
	        if (y > 0) {
	        	if (DEBUG)  {
		            printf("The multiplicative inverse of %lld mod %lld is %lld\n\n", b, a, y);
		            printf("(%lld * %lld) mod %lld = %lld\n\n", b, y, a, b*y % a);
		        }
		        val = y;
		    }
		    else {
		    	if (DEBUG) {
		            printf("The multiplicative inverse of %lld mod %lld is (%lld - %lld) = %lld\n\n", b, a, a, -y, a+y);
		            printf("(%lld * %lld) mod %lld = %lld\n\n", b, a+y, a, (b*(a+y)) % a);
		        }
		        val = a+y;
	        }
	}
	else {
		printf("There is no multiplicative inverse of %lld mod %lld\n\n", num, modulo);
		printf("gcd (%lld, %lld) = %lld\n\n", a, b, gcd);
		printf("gcd != 1 --> %lld and %lld are not relatively prime\n\n", a, b);
	}

	return(val);
}

///////////////////////////////////////////////////////////////////////////////////////////
//
// Functions for Pollard Rho & Van Orchoost & Wiener algorithms
//
///////////////////////////////////////////////////////////////////////////////////////////


long long get_group(POINT_T R, int L)  {
    long long group;

    group = R.x % L;
    return(group);
}


POINT_T nextpoint(POINT_T P, long long a, long long p,
                    long long order, int L, POINT_T *Triplet)
{
    POINT_T Y;
    long long g;

    g = get_group(P, L);
    Y = addpoints(P, Triplet[g], a, p, order);

    return(Y);
}


long long int get_k(long long c1, long long d1, long long c2, long long d2, long long order)
{
    long long  num, den, invden, k;

    num = (c1 - c2);
    while (num < 0)  num = num+order;
    num = num % order;

    den = (d2 - d1);
    while (den < 0)  den = den+order;

    invden = multiplicative_inverse(den, order);
    k = (num*invden) % order;

    return(k);
}

long long get_rand(long long order) {
    long long num = rand() % order;
    return(num);
}

void calc_P_sums(POINT_T P, long long a, long long p, long long order, int nbits) {
    long long i, j;
    POINT_T Ps=P;

    Psums = (POINT_T *) malloc(sizeof(POINT_T) * nbits);

    Psums[0] = Ps;
    for (i=1; i < nbits; i++) {
        Ps = addpoints(Ps, Ps, a, p, 0);
        Psums[i] = Ps;
    }
    return;
}

void calc_Q_sums(POINT_T Q, long long a, long long p, long long order, int nbits) {
    long long i, j;
    POINT_T Qs=Q;

    Qsums = (POINT_T *) malloc(sizeof(POINT_T) * nbits);

    Qsums[0] = Qs;

    for (i=1; i < nbits; i++) {
        Qs = addpoints(Qs, Qs, a, p, 0);
        Qsums[i] = Qs;
    }

    return;
}

void calc_Q (POINT_T *pQ, POINT_T *Psums, long long k, int nbits, long long a, long long p)
{
    unsigned i, bitmask = 1;

    *pQ = EMPTY_POINT;
    bitmask = 1;

    for (i=0; i < nbits; i++) {
        if (k & bitmask) {
            *pQ = addpoints(*pQ, Psums[i], a, p, 0);
        }
        bitmask = (bitmask << 1);
    }

    return;
}



void
rand_itpset(IT_POINT_SET *itpset, POINT_T *Psums, POINT_T *Qsums, int tid,
            long long a, long long p, long long order, int L, int nbits,
            int algorithm)
{
    long long i, j, r, s;
    uint32_t bitmask = 1;

    for (i=0; i < L; i++)
    {
        r = get_rand(order);
        s = get_rand(order);

        itpset->itpoint[i] = EMPTY_POINT;
        bitmask = 1;

        for (j=0; j < nbits; j++) {
            if (r & bitmask) {
                itpset->itpoint[i] = addpoints(itpset->itpoint[i], Psums[j], a, p, 0);
            }
            if (s & bitmask) {
                itpset->itpoint[i] = addpoints(itpset->itpoint[i], Qsums[j], a, p, 0);
            }
            bitmask = (bitmask << 1);
        }

        itpset->itpoint[i].r = r;
        itpset->itpoint[i].s = s;
    }

    return;
}


long long equal(POINT_T X1, POINT_T X2)  {

    if (X1.x != X2.x) return(0);
    if (X1.y != X2.y) return(0);
    if (X1.r != X2.r) return(0);
    if (X1.s != X2.s) return(0);

    return(1);
}



/////////////////////////////////////////////////////////////////////////////
//
// This function attempts to store a new point in the hashtable.
//
/////////////////////////////////////////////////////////////////////////////

void check_slot_and_store(POINT_T X, POINT_T *Y, int token, int *retval)
{
    uint32_t index1 = (uint32_t) X.y, index2 = (uint32_t) ((X.y + 2) >> 1);
    POINT_T H;

    // Modify index2 according to the token value
    if (token > 0) index2 = (index2 + token*index1) % TABLESIZE;

    hashword2((const uint32_t *) &X.x, 1, &index1, &index2);
    index1 = index1 % TABLESIZE;
    index2 = index2 % TABLESIZE;

    // Check slot1; if slot1 is empty, store the point and return
    if (equal(hashtable[index1], EMPTY_POINT)) {
        hashtable[index1] = X;
        *retval = STORED;
        return;
    }
    else {
        // Get the point from slot1
        H = hashtable[index1];

        // Check the point from slot1
        if ((H.x == X.x) && (H.y == X.y))  {
            // If point is the same with different coefficients, return point
            if ((H.r != X.r) || (H.s != X.s)) {
                *Y = H;
                *retval = GOOD_COLLISION;
                return;
            }
        }
        // Point in slot1 was either different or the same with the same coefficients.
        // Check slot2; if slot2 is empty, store the point and return
        if (equal(hashtable[index2], EMPTY_POINT)) {
            hashtable[index2] = X;
            *retval = STORED;
            return;
        }
        else {
            // Get the point from slot2
            H = hashtable[index2];

            // Test the point from slot2
            if ((H.x == X.x) && (H.y == X.y))  {
                // If point in slot 2 is the same with different coefficients, return point
                if ((H.r != X.r) || (H.s != X.s)) {
                    *Y = H;
                    *retval = GOOD_COLLISION;
                    return;
                }
                else
                    // If point is the same with same coefficients report unfruitful
                    *retval = UNFRUITFUL_COLLISION;
                    return;
            }
            else {
                // If point is different report empty slot not found
                *Y = H;
                *retval = EMPTY_SLOT_NOT_FOUND;
                return;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////
//
// This function calculates a new iteration point set NIPS (composed of L points) by
// multiplication a given iteration point set IPS by an integer mult:
//
//                             NIPS = mult * itSetBase
//
///////////////////////////////////////////////////////////////////////////////////////////

void
mult_PQpset2(IT_POINT_SET *nitpset, IT_POINT_SET *ipset, long long mult, POINT_T *Psums,
             POINT_T *Qsums,int L, int nbits, long long a, long long p, long long order)
{
    int i, j;
    long long r, s;
    uint32_t bitmask;

    for (i=0; i < L; i++) {
        bitmask = 1;
        r = (mult * ipset->itpoint[i].r) % order;
        s = (mult * ipset->itpoint[i].s) % order;

        nitpset->itpoint[i] = EMPTY_POINT;
        for (j=0; j < nbits; j++) {
            if (r & bitmask) {
                nitpset->itpoint[i] = addpoints(nitpset->itpoint[i], Psums[j], a, p, 0);
            }
            if (s & bitmask) {
                nitpset->itpoint[i] = addpoints(nitpset->itpoint[i], Qsums[j], a, p, 0);
            }
            bitmask = (bitmask << 1);
        }
        nitpset->itpoint[i].r = r;
        nitpset->itpoint[i].s = s;
    }

    return;
}


///////////////////////////////////////////////////////////////////////////////////////////
//
// This function calculates the iteration point set IPS (composed of L points) for
// a worker by multiplying the iteration point set base (itSetBase) by an integer m:
//
//                             IPS = m * itSetBase
//
///////////////////////////////////////////////////////////////////////////////////////////

void calc_iteration_point_set2(IT_POINT_SET *itpset, IT_POINT_SET *itpsetbase, POINT_T *Psums, POINT_T *Qsums, int tid,
                              long long a, long long p, long long order, int L, int nbits, int algorithm)

{
    int m;

    if (algorithm == VOW) {
        // Each and every worker has its iteration point set equal to itPsetBase
       *itpset = *itpsetbase;
    }

    else if (algorithm == TOHA) {
        if ((tid % 2) == 0) {
            *itpset = *itpsetbase;
        }
        else {
            // Each odd worker's iteration point set is a multiple of itPsetBase (itPsetBase * (tid+3)/2)
            m = (tid+3)/2;
            mult_PQpset2(itpset, itpsetbase, m, Psums, Qsums, L, nbits, a, p, order);
        }
    }

    else if (algorithm == HARES) {
        // Each and every worker's iteration point set is a multiple of itPsetBase (itPsetBase * (tid+1)
        m = tid+1;
        mult_PQpset2(itpset, itpsetbase, m, Psums, Qsums, L, nbits, a, p, order);
    }

    return;
}




///////////////////////////////////////////////////////////////////////////////////////////
//
// This function calculates the iteration point set IPS (composed of L points) for
// a worker by multiplying the iteration point set base (itSetBase) by an integer m:
//
//                             IPS = m * itSetBase
//
///////////////////////////////////////////////////////////////////////////////////////////

void
calc_iteration_point_set3(IT_POINT_SET *itpset, IT_POINT_SET *itpsetbase,
                          POINT_T *Psums, POINT_T *Qsums, int tid, long long a,
                          long long p, long long order, int L, int nbits,
                          int algorithm)
{
    if (algorithm == VOW) {
        // Each and every worker has its iteration point set equal to itPsetBase
       *itpset = *itpsetbase;
    }

    else if (algorithm == TOHA) {
        if ((tid % 2) == 0) {
            *itpset = *itpsetbase;
        }
        else {
            // Each odd worker's iteration point set is randomly calculated
            rand_itpset(itpset, Psums, Qsums, tid, a, p, order, L, nbits, algorithm);
        }
    }

    else if (algorithm == HARES) {
        // Each and every worker's iteration point set is randomly calculated
        rand_itpset(itpset, Psums, Qsums, tid, a, p, order, L, nbits, algorithm);
    }

    return;
}



///////////////////////////////////////////////////////////////////////////////////////////
//
// This function calculates the initial point IP for a worker based on points P and Q:
//
//                                 IP = rP + sQ
//
///////////////////////////////////////////////////////////////////////////////////////////

void initial_point(POINT_T *init_point, POINT_T *Psums, POINT_T *Qsums, int nbits,
                   int gtid, int alg, long long nworkers, long long a,
                   long long p, long long order)
{
    long long i, r, s;
    uint32_t bitmask = 1;
    POINT_T ipoint = { 0, 0, 0, 0 };

    //if ((alg == VOW) || ((alg == TOHA) && (gtid % 2) == 0) || (alg == HARES)) {
    if ((alg == VOW) || (alg == TOHA) || (alg == HARES)) {
        r = get_rand(order);
        s = get_rand(order);

        for (i=0; i < nbits; i++) {
            if (r & bitmask) {
                ipoint = addpoints(ipoint, Psums[i], a, p, 0);
            }
            if (s & bitmask) {
                ipoint = addpoints(ipoint, Qsums[i], a, p, 0);
            }
            bitmask = (bitmask << 1);
        }

        ipoint.r = r;
        ipoint.s = s;

        *init_point = ipoint;
    }
    //else {
    //    *init_point = *(init_point + 1);
    //}

    return;
}



////////////////////////////////////////////////////////////////////////////
//
// This function executes the setup for one worker
//
////////////////////////////////////////////////////////////////////////////

void
setup_worker(POINT_T *X, POINT_T P, POINT_T Q, long long a, long long p, long long order,
             const int L, const int nbits, int id, int alg)
{
    // Clear the iteration point set memory
    memset(&itPsets[id], 0, sizeof(IT_POINT_SET));

    // Calculate the iteration point set for this worker
    calc_iteration_point_set3(&itPsets[id], &itPsetBase, Psums, Qsums, id, a, p, order, L, nbits, alg);

    // Calculate the initial point for this worker
    initial_point(&initX[id], Psums, Qsums, nbits, id, alg, nworkers, a, p, order);
    X[id] = initX[id];

    return;
}


////////////////////////////////////////////////////////////////////////////
//
// This function executes one iteration for one worker
//
////////////////////////////////////////////////////////////////////////////

void
worker_it_task(long long a, long long p, long long order,
               const int L, int id, int *key)
{
    // Calculate the next point
    X[id] = nextpoint(X[id], a, p, order, L, &itPsets[id]);

    // Handle the new point checking if the key has been found
    handle_newpoint(X[id], order, key);
}





/////////////////////////////////////////////////////////////////////////////
//
// This function stores a new point in the hashtable and looks for
// a "good" collision: two points (P1, P2) with equal coordinates
// (X,Y) but with different coefficients (r,s), such that:
//
//      P1 = r1*P + s1*Q,       P2 = r2*P + s2Q,       P1 = P2
//
/////////////////////////////////////////////////////////////////////////////

void handle_newpoint(POINT_T X, long long order, long long *key)
{
    int retval, num_emptyslotnotfound = 0;
    POINT_T H;

    do {
        check_slot_and_store(X, &H, num_emptyslotnotfound, &retval);
        switch (retval)  {
            case GOOD_COLLISION:
                *key = get_k(X.r, X.s, H.r, H.s, order);
                break;
            case EMPTY_SLOT_NOT_FOUND:
                num_emptyslotnotfound++;
                if (num_emptyslotnotfound == MAX_EMPTY_SLOT_NOT_FOUND) {
                    printf("ATTENTION!!! TOO MANY FAILED STORAGE ATTEMPTS -- exiting ...\n");
                    printf("               H = "); printP(H); printf("\n\n");
                    exit(-3);
                }
            case UNFRUITFUL_COLLISION:
                // printf("   --  Unfruitful collision -- continuing ...\n");
                printf("@");
                break;
            case STORED:
                break;
        }

        if ((retval == GOOD_COLLISION) || (retval == UNFRUITFUL_COLLISION)
                                                     || (retval == STORED))  break;

    } while (num_emptyslotnotfound < MAX_EMPTY_SLOT_NOT_FOUND);

    if (retval != GOOD_COLLISION) *key = NO_KEY_FOUND;

    return;
}


int main()
{
	long long a, b, p, maxorder, k;
	long long key;
	int it_number, minits, maxits = 0, run=1;
	POINT_T G, Q, P;
	time_t  init, term;
	double start, end;
	double convergence_time, setup_time;
	double average_it_num = 0.0, average_it_time = 0.0;
	double total_it_num = 0.0, total_it_time = 0.0;
	uint32_t bitmask;
	int i, id;

	system("cls");

	switch (algorithm)  {
        case VOW:   printf("\n\nPOLLARD RHO ALGORITHM  --  VOW\n");                  break;
        case TOHA:  printf("\n\nPOLLARD RHO ALGORITHM  --  Tortoises and Hares\n");  break;
        case HARES: printf("\n\nPOLLARD RHO ALGORITHM  --  Tortoises\n");            break;
	}

	/////////////////////////////////////////////////////////////////////////
	//
	//     ELLIPTIC CURVE EQUATION (WEIERSTRASS FORM)
	//
	//                 Y^2 = X^3 + aX + b
	//
	/////////////////////////////////////////////////////////////////////////


    printf("\nNumber of bits of the EC prime field (16/20/24/28/32): %d\n\n", nbits);

   // Pick elliptic curve parameters for chosen number of bits of the prime field module p
    switch (nbits)  {
        case 16:
            a = 1;   b = 44;         p = 16747;                         // 16 bits
            maxorder = 16931;        P.x = 1; P.y = 5626;
            k = 8047;
            break;
        case 20:
            a = 1;   b = 44;         p = 1048507;                       // 20 bits
            maxorder = 1049101;      P.x = 373173; P.y = 395411;
            k = 3051;
            break;
        case 24:
            a = 1;   b = 44;         p = 16774421;                       // 24 bits
            maxorder = 16770883;     P.x = 4530807; P.y = 1256865;
            k = 2349;
            break;
        case 28:
            a = 1;   b = 44;         p = 268434997;                      // 28 bits
            maxorder = 268446727;    P.x = 47793986; P.y = 101136283;
            k = 7514;
            break;
       case 32:
            a = 1;   b = 44;         p = 4294966981;                      // 32 bits
            maxorder = 4295084473;   P.x = 3437484969; P.y = 579918983;
            k = 2037;
            break;
        default:
            printf("\n\nnbits is not in the defined set (16, 20, 24, 28, 32)\n\n");
            exit(-1);
    }


	if (a == 1) printf("\nElliptic curve:   y^2 = x^3 + x + %lld      (mod %lld, %d bits)\n\n", b, p, nbits);
	else printf("\nElliptic curve:   y^2 = x^3 + %lldx + %lld      (mod %lld, %d bits)\n\n", a, b, p, nbits);

	printf("Order of the curve = %lld\n\n", maxorder);

	printf("Number of workers = %lld\n\n", nworkers);

    // Calculating point Q = kP
    printf("Calculating point Q = kP      (k = %lld)\n", k);
    calc_P_sums(P, a, p, maxorder, nbits);
    calc_Q(&Q, Psums, k, nbits, a, p);
    calc_Q_sums(Q, a, p, maxorder, nbits);

    printf("\n");
    printf("          P = (%8lld, %8lld)          (Base Point)\n", P.x, P.y);
    printf("          Q = (%8lld, %8lld)          (Q = kP    )\n\n\n", Q.x, Q.y);

    // Initialize minits
    minits = maxorder;

    // Loop to calculate the desired secret key (k) MAX_RUNS times, each one
    // with randomly chosen step and starting points for each thread (worker).
    // Remember the number of iterations each run took to find the key (k).
    // Then calculate the expected (average) number of iterations that this
    // calculation takes.

    while (1) {

        // Initialize the number of iterations
        it_number = 1;

        // Initialize the random generator
        srand(time(NULL));

        // Start counting the setup time
        start = get_wall_time();

        // Calculate the iteration point set base (randomly)
        rand_itpset(&itPsetBase, Psums, Qsums, id, a, p, maxorder, L, nbits, algorithm);

        // Set up the running environment for the search for all workers
        printf("Run[%3d] setup:    ", run);
        for (id=0; id < nworkers; id++) {
            setup_worker(X, P, Q, a, p, maxorder, L, nbits, id, algorithm);
            printf("\b\b\b%3d", id+1);
            fflush(stdout);
        }
        printf("\nRun[%3d] iterations: ", run);

        // Stop counting the setup time and calculate it
        end = get_wall_time();
        setup_time = 1000*(double)((end - start) / CLOCKS_PER_SEC);

        // Start counting the execution time
        start = get_wall_time();

        for ( ; ; ) {

            for (id=0; id < nworkers; id++) {
                worker_it_task(a, p, maxorder, L, id, &key);
                if (key != NO_KEY_FOUND) break;  // is the search over?
            }

            if (it_number%10  ==  0) printf(" %d ", it_number);
            else {
                if (wherex() >= 80) printf("\n                     ");
                printf("+");
            }
            fflush(stdout);

            if (key != NO_KEY_FOUND) {
                printf("  (%d its)\n", it_number);
                break;
            }
            it_number++;
        }

        // Recover the current time and calculate the execution time
        end = get_wall_time();
        convergence_time = 1000*(double)((end - start) / CLOCKS_PER_SEC);

        // Choose text for describing the algorithm iteration point sets
        char *stepdef;
        switch (algorithm)  {
            case VOW:   stepdef = "all-equal";                break;
            case TOHA:  stepdef = "1/2 equal + 1/2 varying";  break;
            case HARES: stepdef = "all-varying";              break;
        }

        // Run converged. Print information for it.
        printf("\nRun[%3d] converged after %4d iterations: k = %lld, nworkers = %4d,\n",
               run, it_number, key, nworkers);
        printf("         setup time = %6.1lf s, conv time = %6.1lf s (itfs \"%s\")\n\n",
               setup_time, convergence_time, stepdef);

        // Keep track of the minimum number of iterations needed to converge in all runs.
        if (it_number < minits) {
            minits = it_number;
        }

        if (it_number > maxits) {
            maxits = it_number;
        }

        total_it_num  = total_it_num  + it_number;
        total_it_time = total_it_time + convergence_time;

        // Cleanup the hash table
        for (i=0; i < TABLESIZE; i++) hashtable[i] = EMPTY_POINT;

        if (run == MAX_RUNS) break;
        run++;
    }   // Loop to do various different executions __ while (1)

     // Final statistics
    printf("\n\nFINAL STATISTICS\n\n");
    printf("Runs = %d, Min Its # = %d,  Max Its = %d, Av. Iteration # = %.0lf, ", run, minits, maxits, total_it_num/MAX_RUNS);
    printf("Av. Iteration Time = %.1lf s, Total Time = %.1lf s\n\n", total_it_time/run, total_it_time);
    fflush(stdout);

	exit(0);
}

