#include "acutest.h"
#include "utils.h"


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
*/

int test_data_neighbors[][5] = {
	{ 1, 2, 3 },
	{ 0, 2, 3 },
	{ 3, 4, 1 },
	{ 2, 4, 1 },
	{ 2, 3, 1 }
};

/*
	Reverse neighbors:
	0: 1
	1: 0, 2, 3, 4
	2: 0, 1, 3, 4
	3: 0, 1, 2, 4
	4: 2, 3
*/

void test_create(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	TEST_ASSERT(graph->k == k);
	TEST_ASSERT(graph->points == points);
	TEST_ASSERT(graph->dim == dim);
	TEST_ASSERT(graph->similarity_comparisons == 0);

	KNNGraph_destroy(graph);
}

void test_export_import_graph(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	/* Export graph: N, dim, K, K Neighbors for node[0], K Neighbors for node[1],... */
	KNNGraph export_graph = KNNGraph_create(test_data, euclidean_dist, k, dim, points);
	KNNGraph_bruteforce(export_graph);
	KNNGraph_export_graph(export_graph, "test_data.bin");

	/* Import graph */
	KNNGraph import_graph = KNNGraph_import_graph("test_data.bin", test_data, euclidean_dist);

	/* Check that imported graph is correct */
	TEST_ASSERT(import_graph->k == k);
	TEST_ASSERT(import_graph->dim == dim);
	TEST_ASSERT(import_graph->points == points);
	for (size_t i = 0UL; i < points; ++i) {
		for (size_t j = 0UL; j < k; ++j) {
			TEST_ASSERT(import_graph->neighbors[i][0].id == test_data_neighbors[i][j]);
			vector_delete(import_graph->neighbors[i], 0);
		}
	}
	KNNGraph_destroy(import_graph);
	KNNGraph_destroy(export_graph);

    CHECK_CALL(remove("test_data.bin"), -1);
}

void test_bruteforce(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);

	/* Build KNN graph with exhaustive search and check that the result 
		is the same as the (manually calculated) expected result */

	KNNGraph_bruteforce(graph);
	for (size_t i = 0UL; i < points; ++i) {
		for (size_t j = 0UL; j < k; ++j) {
			TEST_ASSERT(graph->neighbors[i][0].id == test_data_neighbors[i][j]);
			vector_delete(graph->neighbors[i], 0);
		}
	}
	KNNGraph_destroy(graph);
}

void test_recall(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	KNNGraph graph_ = KNNGraph_create(test_data, manhattan_dist, k, dim, points);

	KNNGraph_bruteforce(graph);
	KNNGraph_bruteforce(graph_);

	/* Graphs are the exact same */
	TEST_ASSERT(fabs(KNNGraph_recall(graph, graph_) - 1.0) < 1e-9);

	KNNGraph_destroy(graph);
	KNNGraph_destroy(graph_);
}


void test_nndescent(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	KNNGraph ground_truth = KNNGraph_create(test_data, manhattan_dist, k, dim, points);

	KNNGraph_nndescent(graph, 0.0, 1.0, 0);
	KNNGraph_bruteforce(ground_truth);

	/* Build KNN graph with both exhaustive search and NNDescent and check that their results 
	   are at least 90% similar (with this trivial dataset, they are the exact same) */

	TEST_ASSERT(KNNGraph_recall(graph, ground_truth) > 0.9);

	KNNGraph_destroy(graph);
	KNNGraph_destroy(ground_truth);
}

void test_knearest(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	KNNGraph_bruteforce(graph);

	for (size_t i = 0UL; i < points; ++i) {
		Neighbor *neighbors = KNNGraph_KNearest(graph, test_data[i]);
		for (size_t j = 0UL; j < k; ++j)	
			TEST_ASSERT(neighbors[j].id == test_data_neighbors[i][j]);
		free(neighbors);
	}
	KNNGraph_destroy(graph);
	
}

TEST_LIST = {
	{ "test_create",			  test_create              },
	{ "test_export_import_graph", test_export_import_graph },
	{ "test_bruteforce", 		  test_bruteforce          },
	{ "test_recall",			  test_recall              },
	{ "test_nndescent", 		  test_nndescent           },
	{ "test_knearest",			  test_knearest            },
	{  NULL,                      NULL                     }
};
