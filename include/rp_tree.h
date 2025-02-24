#ifndef RP_TREE_H
#define RP_TREE_H

#include "vector.h"

struct node {
    uint32_t left_ind;
    uint32_t right_ind;
    uint32_t *data;
    float *hyperplane;
    float offset;
};

typedef struct rp_tree {
    struct node *nodes;
    float **data;
    uint32_t leaf_size;
    uint32_t points;
    uint32_t dimension;
    unsigned int seed;
} *RPTree;

RPTree RPTree_create(float **data, uint32_t points, uint32_t dimension, uint32_t leaf_size);

RPTree *RPTree_build_forest(uint32_t n_trees, float **data, uint32_t points, 
                            uint32_t dimension, uint32_t leaf_size);

void RPTree_split(RPTree tree);

void RPTree_destroy(RPTree tree);


#endif /* RP_TREE_H */