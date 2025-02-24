#include "acutest.h"
#include "utils.h"
#include "bitset.h"


// static bool cmp_pairs(Pair a, Pair b) {
// 	return a.u1 == b.u1 && a.u2 == b.u2;
// }



float *test_data[5] = {	
	(float[]){ 1.0, 1.0 },
	(float[]){ 2.0, 2.0 },
	(float[]){ 3.0, 3.0 },
	(float[]){ 3.0, 4.0 },
	(float[]){ 4.0, 3.0 }
};

/*  For above data: (using manhattan distance metric)
	Neighbors are:
	0: 1, 2, 3
	1: 0, 2, 3
	2: 3, 4, 1
	3: 2, 4, 1
	4: 2, 3, 1 

    As such, reverse neighbors are:
    0: 1
    1: 0, 2, 3, 4
    2: 0, 1, 3, 4
    3: 0, 1, 2, 4
    4: 2, 3
*/

void test_reverse_neighbors(void) {
	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, 3, 2, 5);

	KNNGraph_bruteforce(graph);
	Neighbor **rev = get_reverse_neighbors(graph);

	TEST_ASSERT(rev[0][0].id == 1);
	TEST_ASSERT(vector_size(rev[0]) == 1);


	uint32_t rev1[] = { 0, 2, 3, 4 };
	for (size_t i = 0Ul; i < 4; ++i)
		TEST_ASSERT(rev[1][i].id == rev1[i]);


	uint32_t rev2[] = { 0, 1, 3, 4 };
	for (size_t i = 0UL; i < 4; ++i)
		TEST_ASSERT(rev[2][i].id == rev2[i]);


	uint32_t rev3[] = { 0, 1, 2, 4 };
	for (size_t i = 0UL; i < 4; ++i)
		TEST_ASSERT(rev[3][i].id == rev3[i]);


	uint32_t rev4[] = { 2, 3 };
	for (size_t i = 0UL; i < 2; i++)
		TEST_ASSERT(rev[4][i].id == rev4[i]);


	for (int i = 0; i < graph->points; i++)
		vector_destroy(rev[i]);
	free(rev);

	KNNGraph_destroy(graph);
}

void test_graph_init(void) {
	uint32_t points = 10;
	uint32_t dim = 3;
	uint32_t k = 5;


	float **data = malloc(points * sizeof(float*));
	for (uint32_t i = 0; i < points; ++i) {
		data[i] = malloc(dim * sizeof(float));
		for (uint32_t j = 0; j < dim; ++j)
			data[i][j] = (float) i;
	}

	KNNGraph graph = KNNGraph_create(data, manhattan_dist, k, dim, points);

	/* Initialize graph with graph_init and check that a valid KNN graph is produced */
	graph_init(graph);
	for (size_t i = 0UL; i < points; ++i) {
		TEST_ASSERT(vector_size(graph->neighbors[i])== k);
		for (size_t j = 0UL; j < k; ++j) {
			uint32_t neighbor_id = graph->neighbors[i][j].id;
			/* Node can't be a neighbor to itself */
			TEST_ASSERT(neighbor_id != i);

			/* Check for duplicates */
			for (size_t z = 0UL; z < k; ++z) {
				if (z == j)
					continue;
				TEST_ASSERT(graph->neighbors[i][z].id != neighbor_id);
			}
		}
	}

    for (size_t i = 0UL; i < points; ++i)
        free(data[i]);
    free(data);

	KNNGraph_destroy(graph);
}


void test_sample(void) {
	uint32_t n = 20;
	float rate = 0.1;
	
	uint32_t *vec = vector_create(uint32_t, VEC_MIN_CAP);
	for (uint32_t i = 0; i < n; ++i)
		vector_insert(vec, i);
	
	sample(&vec, (uint32_t)floor((float)n * rate), 0);
	TEST_ASSERT(vector_size(vec) == (uint32_t)floor((float)n * rate));

	/* Check that there are no duplicates */
	uint8_t *bs = BITSET_CREATE(n);
	for (size_t i = 0UL; i < vector_size(vec); ++i) {
		TEST_ASSERT(!BITSET_CHECK(bs, vec[i]));
		BITSET_SET(bs, vec[i]);
	}
	free(bs);
	vector_destroy(vec);
}

void test_collect_pairs(void) {
	uint32_t old[] = { 1, 2, 3 };
	uint32_t new[] = { 4, 5, 6 };
	uint32_t sets[][5] = {
		{ 5, 6, 1, 2, 3 },
		{ 6, 1, 2, 3, 0 },
		{ 1, 2, 3, 0, 0 }	
	};

	// expected pairs
	// new_id     sets
	// { 4, { 5, 6, 1, 2, 3 } }
	// { 5, { 6, 1, 2, 3 } }
	// { 6, { 1, 2, 3 } }

	uint32_t *old_ = vector_create(uint32_t, VEC_MIN_CAP);
	uint32_t *new_ = vector_create(uint32_t, VEC_MIN_CAP);
	for (size_t i = 0UL; i < ARRAY_SIZE(new); ++i)
		vector_insert(new_, new[i]);

	for (size_t i = 0UL; i < ARRAY_SIZE(old); ++i)
		vector_insert(old_, old[i]);


	Pair *pairs = collect_pairs(old_, new_);
	TEST_ASSERT(pairs != NULL);

	for (size_t i = 0UL; i < ARRAY_SIZE(new); ++i) {
		TEST_ASSERT(pairs[i].id == new[i]);
		for (int j = 0; j < vector_size(pairs[i].neighbors); ++j)
			TEST_ASSERT(pairs[i].neighbors[j].id == sets[i][j]);
	}
	vector_destroy(new_);
	vector_destroy(old_);

	for (size_t i = 0UL; i < ARRAY_SIZE(new); ++i)
		vector_destroy(pairs[i].neighbors);
	free(pairs);
}

void test_euclidean(void) {
	float a[] = {0.123, -0.12,  0.43, 0.53, 0.19, 0.0};
	float b[] = {-0.21, -0.125, 0.8, 0.235, -0.213, 0.0};
	uint32_t dim = ARRAY_SIZE(a) - 1;

	a[dim] = dot_product(a, a, dim);
	b[dim] = dot_product(b, b, dim);

	float dist = euclidean_dist(a, b, dim);
	float opt_dist = sqrt(optimized_euclidean(a, b, dim));

	TEST_ASSERT(fabs(dist - opt_dist) < 1e-7);
}



TEST_LIST = {
    { "test_graph_init",        test_graph_init        },
    { "test_reverse_neighbors", test_reverse_neighbors },
	{ "test_sample",			test_sample            },
	{ "test_collect_pairs",		test_collect_pairs     },
	{ "test_euclidean",         test_euclidean         },
    {  NULL,                    NULL                   }
};
