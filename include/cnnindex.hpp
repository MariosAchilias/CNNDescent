extern "C" {
    #include "knngraph.h"
    #include "utils.h"
}

enum Dist {
    EUCLIDEAN,
    MANHATTAN
};

#include <vector>
#include <tuple>

class CNNIndex {
public:
    int k;
    int dim;
    KNNGraph graph;
    CNNIndex(int k, int dim, enum Dist dist) : k(k), dim(dim) {graph = KNNGraph_create(NULL, dist == EUCLIDEAN ?  euclidean_dist : manhattan_dist, k, dim, 0);};
    CNNIndex(char *filename, enum Dist dist) {KNNGraph_import_graph(filename, NULL, dist == EUCLIDEAN ? euclidean_dist : manhattan_dist);};
    ~CNNIndex() {KNNGraph_destroy(graph);};
    void save(char *filename) {KNNGraph_export_graph(graph, filename);};
    void set_n_threads(int n) {n_threads = n;};
    void add_point(float *point) {KNNGraph_add_point(graph, point);};
    void build_index_bruteforce() {KNNGraph_bruteforce(graph);};
    void build_index_nndescent(float precision, float sample_rate, int n_trees) {KNNGraph_nndescent(graph, precision, sample_rate, n_trees);};
    std::vector<std::tuple<int, float>> get_k_nearest(float *point) {
        Neighbor *knn = KNNGraph_KNearest(graph, point);
        std::vector<std::tuple<int, float>> knn_;
        for (uint32_t i = 0; i < graph->k; i++) {
            knn_.push_back(std::tuple<int, float>{knn[i].id, knn[i].dist});
        }
        free(knn);
        return knn_;
    };
    std::vector<std::tuple<int, float>> get_k_nearest(int i) {
        Neighbor *knn = graph->neighbors[i];
        std::vector<std::tuple<int, float>> knn_;
        for (uint32_t i = 0; i< graph->k; i++) {
            knn_.push_back(std::tuple<int, float>{knn[i].id, knn[i].dist});
        }
        return knn_;
    };
 
};