#ifndef COMMON_H
#define COMMON_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/mman.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <ctype.h>
#include <omp.h>


#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))


extern uint32_t n_threads;

typedef struct {
	uint32_t id;
	float dist;
	bool flag;
} Neighbor;


typedef struct {
	uint32_t id;
	Neighbor *neighbors;
} Pair;

typedef int (*Comparator)(Neighbor a, Neighbor b);
typedef float (*DistanceFunc)(float const *a, float const *b, uint32_t dim);


typedef struct {
	uint32_t k;
	uint32_t points;
	uint32_t dim;
	uint32_t n_trees;
	bool big_endian;
	const char *file;
	float *data;
	DistanceFunc metric;    
	float sample_rate;
	float precision;
} KnnArgs;


#define CHECK_CALL(call, error_value)                                                   \
	do {                                                                                \
		if ((call) == error_value) {                                                    \
			fprintf(stderr,                                                             \
				"Call %s failed with an error message: (%s) at file %s at line %d\n",   \
				#call, strerror(errno), __FILE__, __LINE__);                            \
			exit(EXIT_FAILURE);                                                         \
		}                                                                               \
	} while (0);



#define CALC_TIME(call) ({								\
	struct timeval start, end;							\
	CHECK_CALL(gettimeofday(&start, NULL), -1);			\
	(call);												\
	CHECK_CALL(gettimeofday(&end, NULL), -1);			\
	((double)(end.tv_sec - start.tv_sec) +				\
	         (end.tv_usec - start.tv_usec) / 1e6);		\
})


#define MIN(a, b) ({	\
	uint32_t a_ = a;	\
	uint32_t b_ = b;	\
	a_ > b_ ? b_ : a_;	\
})


#endif /* COMMON_H */
