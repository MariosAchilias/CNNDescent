#ifndef KNN_GRAPH_H
#define KNN_GRAPH_H

#include "vector.h"


struct KNNGraph_ {
	uint32_t k;
	uint32_t dim;
	uint32_t points;
	float **data;
	Neighbor **neighbors;
	DistanceFunc dist;
	uint32_t similarity_comparisons;
};

typedef struct KNNGraph_* KNNGraph;

KNNGraph KNNGraph_create(float **data, DistanceFunc dist, uint32_t k, 
	                     uint32_t dim, uint32_t points);

void KNNGraph_add_point(KNNGraph graph, float *point);

void KNNGraph_bruteforce(KNNGraph graph);

void KNNGraph_nndescent(KNNGraph graph, float precision, float sample_rate, uint32_t n_trees);

Neighbor *KNNGraph_KNearest(KNNGraph graph, float *point);

float KNNGraph_recall(KNNGraph graph, KNNGraph ground_truth);

void KNNGraph_destroy(KNNGraph graph);

KNNGraph KNNGraph_import_graph(char *graph_file, float **data, DistanceFunc dist);

void KNNGraph_export_graph(KNNGraph graph, const char *filename);

#endif /* KNN_GRAPH_H */