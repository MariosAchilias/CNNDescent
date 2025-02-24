#include "rp_tree.h"
#include "utils.h"

#define EPSILON 1e-4

RPTree RPTree_create(float **data, uint32_t points, 
                     uint32_t dimension, uint32_t leaf_size) {
    RPTree tree = malloc(sizeof(*tree));
    tree->leaf_size = leaf_size;
    tree->dimension = dimension;
    tree->data = data;
    tree->points = points;
    tree->nodes = vector_create(struct node, VEC_MIN_CAP);
    tree->seed = omp_get_thread_num();

    return tree;
}

static uint32_t tree_split_rec(RPTree tree, uint32_t *data, uint32_t depth) {
    if (depth == tree->leaf_size - 1 && vector_size(data) > tree->leaf_size)
        vector_set_size(data, tree->leaf_size);

    if (vector_size(data) <= tree->leaf_size) {
        /* Leaf nodes have no hyperplane */
        vector_insert(tree->nodes, ((struct node){.hyperplane = NULL, 
                                                  .data = data}));
        return vector_size(tree->nodes) - 1;
    }
    uint32_t *left = vector_create(uint32_t, VEC_MIN_CAP);
    uint32_t *right = vector_create(uint32_t, VEC_MIN_CAP);

    /* Choose a hyperplane using 2 unique data points */
    float offset = 0.0;
    float *hyperplane = malloc(tree->dimension * sizeof(float));
    for (size_t i = 0UL; i < tree->dimension; ++i) {
        int ind = rand_r(&tree->seed) % tree->points;
        int ind_ = rand_r(&tree->seed) % tree->points;
        while (ind == ind_)
            ind_ = rand_r(&tree->seed) % tree->points;
        
        hyperplane[i] = tree->data[ind][i] - tree->data[ind_][i];
        offset += hyperplane[i] * (tree->data[ind][i] + tree->data[ind_][i]) / 2;
    }

    for (size_t i = 0; i < vector_size(data); i++) {
        /* Split data according to their projection onto the hyperplane */
        float margin = dot_product(hyperplane, tree->data[data[i]], tree->dimension) - offset;
        if (margin < -EPSILON) {
            vector_insert(left, data[i]);
        } else if (margin > EPSILON) {
            vector_insert(right, data[i]);
        } else {
            if (rand_r(&tree->seed) % 2 == 0) {
                vector_insert(left, data[i]);
            } else {
                vector_insert(right, data[i]);
            }
        }
    }

    /* If all points fall into one side, split the data randomly 
     * This is useful for data not well suited to hyperplane splitting
     * (e.g. data points very close to each other)
     */
    if (!vector_size(left) || !vector_size(right)) {
        vector_set_size(left, 0);
        vector_set_size(right, 0);
        for (size_t i = 0UL; i < vector_size(data); ++i) {
            if (rand_r(&tree->seed) % 2 == 0) {
                vector_insert(left, data[i]);
            } else {
                vector_insert(right, data[i]);
            }
        }
    }
    vector_destroy(data);

    uint32_t left_ind = tree_split_rec(tree, left, depth + 1);
    uint32_t right_ind = tree_split_rec(tree, right, depth + 1);

    /* Non-leaf nodes store no data */
    vector_insert(tree->nodes, ((struct node){.left_ind = left_ind,
                                              .right_ind = right_ind,
                                              .hyperplane = hyperplane,
                                              .data = NULL,
                                              .offset = offset}));

    return vector_size(tree->nodes) - 1;
}

void RPTree_split(RPTree tree) {
    uint32_t *data = vector_create(uint32_t, tree->points);
    for (size_t i = 0UL; i < tree->points; ++i)
        vector_insert(data, i);
    
    tree_split_rec(tree, data, 0);
}

void RPTree_destroy(RPTree tree) {
    for (size_t i = 0UL; i < vector_size(tree->nodes); ++i) {
        if (tree->nodes[i].data != NULL)
            vector_destroy(tree->nodes[i].data); 
        free(tree->nodes[i].hyperplane);
    }
    vector_destroy(tree->nodes);
    free(tree);
}


RPTree *RPTree_build_forest(uint32_t n_trees, float **data, uint32_t points,
                           uint32_t dimension, uint32_t leaf_size) {
    RPTree *forest = malloc(sizeof(RPTree) * n_trees);
    # pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0UL; i < n_trees; ++i) {
        forest[i] = RPTree_create(data, points, dimension, leaf_size);
        forest[i]->seed += i + time(NULL);
        RPTree_split(forest[i]);
    }

    return forest;
}