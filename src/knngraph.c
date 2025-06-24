#include "knngraph.h"
#include "bitset.h"
#include "utils.h"



KNNGraph KNNGraph_create(float *data, DistanceFunc dist, uint32_t k, 
                         uint32_t dim, uint32_t points) {                      
	KNNGraph graph;
    CHECK_CALL(graph = malloc(sizeof(*graph)), NULL);
	
	graph->neighbors = vector_create(Neighbor*, VEC_MIN_CAP);
	for (size_t i = 0UL; i < points; i++) {
		Neighbor *tmp = malloc(k * sizeof(Neighbor));
		vector_insert(graph->neighbors, tmp);
	}
	
	graph->k = k;
	graph->dim = dim;
	graph->data = data != NULL ? data : vector_create(float, VEC_MIN_CAP);
	graph->dist = dist;
	graph->points = points;

	return graph;
}

void KNNGraph_add_point(KNNGraph graph, float *point) {
	Neighbor *tmp = malloc(sizeof(Neighbor*));
	vector_insert(graph->neighbors, tmp);

	for (uint32_t i = 0; i < graph->dim; i++) {
		vector_insert(graph->data, point[i]);
	}

	vector_insert(graph->data, dot_product(point, point, graph->dim));
	graph->points++;
}

/* Build graph by exhaustively computing distances between all pairs of points  */
void KNNGraph_bruteforce(KNNGraph graph) {
	for (size_t i = 0UL; i < graph->points; ++i) {
		graph->neighbors[i] = vector_create(Neighbor, graph->k);
		omp_init_lock(vector_lock(graph->neighbors[i]));
	}

	# pragma omp parallel for num_threads(n_threads)
	for (size_t i = 0UL; i < graph->points; ++i) {
		for (size_t j = i + 1; j < graph->points; ++j) {
			float dist = graph->dist(&graph->data[i * (graph->dim + 1)], &graph->data[j * (graph->dim + 1)], graph->dim);

			omp_set_lock(vector_lock(graph->neighbors[i]));
			vector_sorted_insert(graph->neighbors[i], ((Neighbor){.id = j, .dist = dist}));
			omp_unset_lock(vector_lock(graph->neighbors[i]));

			omp_set_lock(vector_lock(graph->neighbors[j]));
			vector_sorted_insert(graph->neighbors[j], ((Neighbor){.id = i, .dist = dist}));
			omp_unset_lock(vector_lock(graph->neighbors[j]));
		}
	}
}

static void init_temps(uint32_t ***old_, uint32_t ***new_, uint32_t ***old_r_, uint32_t ***new_r_, uint32_t points, uint32_t k);
static void cleanup_temps(uint32_t **old, uint32_t **new, uint32_t **old_r, uint32_t **new_r, uint32_t points);

void KNNGraph_nndescent(KNNGraph graph, float precision, float sample_rate, uint32_t n_trees) {	
	if (n_trees)
		graph_init_rp(graph, n_trees);
	else
		graph_init(graph);

	uint32_t **old, **new, **old_r, **new_r;
	init_temps(&old, &new, &old_r, &new_r, graph->points, graph->k);

    uint32_t c;
	do {
		c = 0;

		collect_sets(graph, old, new, sample_rate);
		collect_reverse(graph, old, new, old_r, new_r, sample_rate);
		# pragma omp parallel for num_threads(n_threads)\
			reduction(+:c), schedule(nonmonotonic:dynamic, 100)
		for (size_t v = 0UL; v < graph->points; ++v) {
			NodeUpdates *updates = collect_updates(graph, old[v], new[v]);

			c += apply_updates(graph, updates);
		}

	} while (c > precision * ((float) graph->points) * graph->k); /* Early termination */

	cleanup_temps(old, new, old_r, new_r, graph->points);

}

/* Return indices of K nearest points */
Neighbor *KNNGraph_KNearest(KNNGraph graph, float *point) {
	Neighbor *KNearest = vector_create(Neighbor, graph->k);

	uint8_t *candidates_inserted;
	CHECK_CALL(candidates_inserted = BITSET_CREATE(graph->points), NULL);
	Neighbor *candidates = vector_create(Neighbor, graph->k);

	Neighbor **graph_neighbors = get_reverse_neighbors(graph);

	uint32_t start_node = rand() % graph->points;
	float dist = graph->dist(&graph->data[start_node * (graph->dim + 1)], point, graph->dim);

	vector_insert(candidates, ((Neighbor){.id = start_node, .dist = dist}));
	BITSET_SET(candidates_inserted, start_node);

	/* In case starting node has distance of zero to search point (i.e. it is the same point), don't return it */
	if (dist > 1e-9)
		vector_insert(KNearest, ((Neighbor){.id = start_node, .dist = dist}));
	
	while (vector_size(candidates) > 0) {
		uint32_t cand = candidates[0].id;
		vector_delete(candidates, 0);

		vector_append(graph_neighbors[cand], graph->neighbors[cand], graph->k);
		for (size_t i = 0UL; i < vector_size(graph_neighbors[cand]); ++i) {
			uint32_t neighbor = graph_neighbors[cand][i].id;

			if (BITSET_CHECK(candidates_inserted, neighbor))
				continue;

			float dist = graph->dist(&graph->data[neighbor * (graph->dim + 1)], point, graph->dim);

			vector_sorted_insert(candidates, ((Neighbor) {.id = neighbor, .dist = dist}));
			BITSET_SET(candidates_inserted, neighbor);

			/* Don't return search node, if it is contained in the dataset */
			if (dist > 1e-9)
				vector_sorted_insert(KNearest, ((Neighbor) {.id = neighbor, .dist = dist}));
		}
	}

	Neighbor *sorted = malloc(graph->k * sizeof(Neighbor));
	memcpy(sorted, KNearest, graph->k * sizeof(Neighbor));
	vector_destroy(KNearest);
	vector_destroy(candidates);

	for (size_t i = 0UL; i < graph->points; i++)
		vector_destroy(graph_neighbors[i]);	
	free(graph_neighbors);

	free(candidates_inserted);

	return sorted;
}

/* Import already computed graph from file (usable for searching and evaluation) */
KNNGraph KNNGraph_import_graph(char *graph_file, float *data, DistanceFunc dist) {
    int fd;
	CHECK_CALL(fd = open(graph_file, O_RDONLY), -1);

	uint32_t n, k, dim;
	CHECK_CALL(read(fd, &n, sizeof(uint32_t)), -1);
	CHECK_CALL(read(fd, &dim, sizeof(uint32_t)), -1);
	CHECK_CALL(read(fd, &k, sizeof(uint32_t)), -1);


	KNNGraph import_graph = KNNGraph_create(data, dist, k, dim, n);
	for (uint32_t i = 0; i < n; ++i) {
        Neighbor entry = {0};
		import_graph->neighbors[i] = vector_create(Neighbor, import_graph->k);
		for (uint32_t j = 0; j < k; ++j) {
			CHECK_CALL(read(fd, &entry.id, sizeof(uint32_t)), -1);
			CHECK_CALL(read(fd, &entry.dist, sizeof(float)), -1);
			vector_sorted_insert(import_graph->neighbors[i], entry);
		}
	}
	CHECK_CALL(close(fd), -1);
	return import_graph;
}


/* Export the info of the graph to a binary file (first bytes: n, dim, k) */
void KNNGraph_export_graph(KNNGraph export_graph, const char *filename) {
	int fd;
	CHECK_CALL(fd = open(filename, O_CREAT | O_WRONLY, 
	                     S_IRWXU | S_IRWXG | S_IRWXO), -1);

    CHECK_CALL(write(fd, &export_graph->points, sizeof(uint32_t)), -1);
    CHECK_CALL(write(fd, &export_graph->dim, sizeof(uint32_t)), -1);
    CHECK_CALL(write(fd, &export_graph->k, sizeof(uint32_t)), -1);

	for (size_t i = 0UL; i < export_graph->points; ++i) {
        Neighbor *entries = export_graph->neighbors[i];
		for (size_t j = 0UL; j < export_graph->k; j++) {
            CHECK_CALL(write(fd, &entries[j].id, sizeof(uint32_t)), -1);
            CHECK_CALL(write(fd, &entries[j].dist, sizeof(float)), -1);
		}
	}
	CHECK_CALL(close(fd), -1);
}


float KNNGraph_recall(KNNGraph graph, KNNGraph ground_truth) {
	float recall = 0;
	/* Calculate recall for each data point and return the average */
	for (size_t i = 0UL; i < graph->points; ++i) {
		float count = 0;
       for (size_t j = 0UL; j < graph->k; ++j) {
			int ind = vector_find(ground_truth->neighbors[i],
								 cmp_ids, graph->neighbors[i][j]);
			count += ind >= 0;
		}
		recall += count / graph->k;
	}
	return recall / graph->points;
}


void KNNGraph_destroy(KNNGraph graph) {
	for (size_t i = 0UL; i < graph->points; ++i)
		if (graph->neighbors[i] != NULL)
			vector_destroy(graph->neighbors[i]);

	vector_destroy(graph->neighbors);
	free(graph);
}


static void init_temps(uint32_t ***old_, uint32_t ***new_, uint32_t ***old_r_, uint32_t ***new_r_, uint32_t points, uint32_t k) {
	uint32_t **old = malloc(points * sizeof(*old));
	uint32_t **new = malloc(points * sizeof(*new));
	uint32_t **old_r = malloc(points * sizeof(*old));
	uint32_t **new_r = malloc(points * sizeof(*new));

	for (size_t v = 0UL; v < points; ++v) {
		old[v] = vector_create(uint32_t, k);
		new[v] = vector_create(uint32_t, k);
		old_r[v] = vector_create(uint32_t, k);
		new_r[v] = vector_create(uint32_t, k);
	}

	*old_ = old;
	*new_ = new;
	*new_r_ = new_r;
	*old_r_ = old_r;
}

static void cleanup_temps(uint32_t **old, uint32_t **new, uint32_t **old_r, uint32_t **new_r, uint32_t points) {
	for (size_t v = 0UL; v < points; ++v) {
		vector_destroy(old[v]);
		vector_destroy(new[v]);
		vector_destroy(old_r[v]);
		vector_destroy(new_r[v]);
	}

	free(old);
	free(new);
	free(new_r);
	free(old_r);
}