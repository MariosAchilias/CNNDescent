#ifndef UTILS_H
#define UTILS_H

#include "knngraph.h"
#include "vector.h"

void import_dataset(KnnArgs *args);

int cmp_ids(Neighbor a, Neighbor b);

bool parse_args(int argc, const char **argv, KnnArgs *args);

float manhattan_dist(float *f1, float *f2, uint32_t dim);

float euclidean_dist(float *f1, float *f2, uint32_t dim);

void graph_init(KNNGraph graph);

void graph_init_rp(KNNGraph graph, uint32_t n_trees);

Neighbor **get_reverse_neighbors(KNNGraph graph);

void sample(uint32_t **vec, uint32_t size, uint32_t seed);

void collect_sets(KNNGraph graph, uint32_t **old, uint32_t **new_, float sample_rate);

void get_reverse(KNNGraph graph, uint32_t **old, uint32_t **new_, uint32_t **old_r, uint32_t **new_r, float sample_rate);

Neighbor *get_knearest(KNNGraph graph, float *point);

Pair *collect_pairs(uint32_t *old, uint32_t *new_);

float optimized_euclidean(float *f1, float *f2, uint32_t dim);

float dot_product(float *f1, float *f2, uint32_t dim);

#endif /* UTILS_H */
