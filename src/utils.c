#include "utils.h"
#include "bitset.h"
#include "rp_tree.h"

// static void reverse_bytes(char *bytes);
static bool str_to_float(const char *s, float *num);
static bool str_to_int(const char *s, uint32_t *num);

uint32_t n_threads = 1;

void import_dataset(KnnArgs *args) {
	int fd;
	char *file;
	struct stat buffer;	
	uint32_t offset = (!args->big_endian) * sizeof(uint32_t);
	uint32_t size = sizeof(float) * args->dim;


	CHECK_CALL(fd = open(args->file, O_RDONLY), -1);
	CHECK_CALL(fstat(fd, &buffer), -1);
	CHECK_CALL(file = mmap(NULL, buffer.st_size, 
	                       PROT_READ | PROT_WRITE, 
						   MAP_PRIVATE, fd, 0), MAP_FAILED);

	uint32_t numbers = buffer.st_size / sizeof(uint32_t);
	args->points = (numbers - !args->big_endian) / args->dim;

	CHECK_CALL(args->data = malloc(sizeof(float) * (args->dim + 1) * args->points), NULL);
	for (size_t i = 0UL; i < args->points; ++i) {
		uint32_t loc = i * (args->dim + 1);
		memcpy(&args->data[loc], file + offset, size);
		args->data[loc + args->dim] = dot_product(&args->data[loc], &args->data[loc], args->dim);
		file += size;
	}

	CHECK_CALL(munmap(file - args->points * size, buffer.st_size), -1);
	CHECK_CALL(close(fd), -1);
}


/* Used to parse arguments in knn_app (format described in README) */
bool parse_args(int argc, const char **argv, KnnArgs *args) {
	for (size_t i = 0UL; i < argc - 1; ++i) {
		if (strcmp("-file", argv[i + 1]) == 0) {
			args->file = argv[i + 2];
		} else if (strcmp("-k", argv[i + 1]) == 0) {
			if (!str_to_int(argv[i + 2], &args->k))
				return false;   
		} else if (strcmp("-dim", argv[i + 1]) == 0) {
			if (!str_to_int(argv[i + 2], &args->dim))
				return false;
		} else if (strcmp("-endian", argv[i + 1]) == 0) {
			args->big_endian = strcmp(argv[i + 2], "big") == 0;
		} else if (strcmp("-metric", argv[i + 1]) == 0) {
			args->metric = !strcmp(argv[i + 2], "euclidean")     ? euclidean_dist      :      
			               !strcmp(argv[i + 2], "euclidean_opt") ? optimized_euclidean :
						   manhattan_dist;
		} else if (strcmp("-sample", argv[i + 1]) == 0) {
			if (!str_to_float(argv[i + 2], &args->sample_rate))
				return false;
		} else if (strcmp("-precision", argv[i + 1]) == 0) {
			if (!str_to_float(argv[i + 2], &args->precision))
				return false;
		} else if (strcmp("-threads", argv[i + 1]) == 0) {
			if (!str_to_int(argv[i + 2], &n_threads))
				return false;
		} else if (strcmp("-trees", argv[i + 1]) == 0) {
			if (!str_to_int(argv[i + 2], &args->n_trees))
				return false;
		}
	}
	return args->metric && args->k && args->dim && args->file;
}


int cmp_ids(Neighbor a, Neighbor b) {
	return a.id - b.id;
}

float manhattan_dist(float const *f1, float const *f2, uint32_t dim) {
	float dist = 0;
	for (size_t i = 0UL; i < dim; ++i)
		dist += fabs(f1[i] - f2[i]);
	return dist;
}


float euclidean_dist(float const *f1, float const *f2, uint32_t dim) {
	double dist = 0;
	for (size_t i = 0UL; i < dim; ++i) {
		double res = f1[i] - f2[i];
		dist += res * res;
	}
	return sqrt(dist);
}


float optimized_euclidean(float const *f1, float const *f2, uint32_t dim) {
	return f1[dim] + f2[dim] - 2.0 * dot_product(f1, f2, dim);
}


void graph_init_rp(KNNGraph graph, uint32_t n_trees) {

	RPTree *forest = RPTree_build_forest(n_trees, graph->data, 
										 graph->points, graph->dim, 
										 graph->k);

	for (size_t i = 0UL; i < graph->points; ++i) {
		graph->neighbors[i] = vector_create(Neighbor, graph->k);
		omp_init_lock(vector_lock(graph->neighbors[i]));
	}

	# pragma omp parallel for num_threads(n_threads)
	for (size_t tree_i = 0UL; tree_i < n_trees; ++tree_i) {
		for (size_t i = 0UL; i < vector_size(forest[tree_i]->nodes); ++i) {
			if (forest[tree_i]->nodes[i].data == NULL)
				continue;	

			for (size_t j = 0UL; j < vector_size(forest[tree_i]->nodes[i].data); ++j) {
				uint32_t u1 = forest[tree_i]->nodes[i].data[j];
				for (size_t k = j + 1; k < vector_size(forest[tree_i]->nodes[i].data); ++k) {
					uint32_t u2 = forest[tree_i]->nodes[i].data[k];

					float dist = graph->dist(&graph->data[u1 * (graph->dim + 1)],
											 &graph->data[u2 * (graph->dim + 1)],
											 graph->dim);


					omp_set_lock(vector_lock(graph->neighbors[u1]));
					vector_sorted_insert(graph->neighbors[u1], 
					                    ((Neighbor){.id = u2, .dist = dist, .flag = true}));
					omp_unset_lock(vector_lock(graph->neighbors[u1]));

					omp_set_lock(vector_lock(graph->neighbors[u2]));
					vector_sorted_insert(graph->neighbors[u2], 
					                     ((Neighbor){.id = u1, .dist = dist, .flag = true}));
					omp_unset_lock(vector_lock(graph->neighbors[u2]));

				}
			}
		}
		RPTree_destroy(forest[tree_i]);
	}
	free(forest);
}

/* Initialize a graph randomly, with K edges per vertex */
void graph_init(KNNGraph graph) {
	for (size_t i = 0UL; i < graph->points; ++i) {
		graph->neighbors[i] = vector_create(Neighbor, graph->k);
		omp_init_lock(vector_lock(graph->neighbors[i]));
	}

	# pragma omp parallel for simd num_threads(n_threads)
	for (size_t i = 0UL; i < graph->points; ++i) {
		while (vector_size(graph->neighbors[i]) < graph->k) {
			uint32_t num = rand() % graph->points;
			while (num == i)
				num = rand() % graph->points;
			
			float dist = graph->dist(&graph->data[i * (graph->dim + 1)],
									 &graph->data[num * (graph->dim + 1)],
									 graph->dim);

			vector_sorted_insert(graph->neighbors[i], 
								((Neighbor){.id = num, .dist = dist, .flag = true}));
			
		}
	}
}


/*  Build reverse neighbors sets for each vertex by
 *  scanning all outcoming edges of each vertex and
 *  add it to correspoding reverse neighbor sets
 */
Neighbor **get_reverse_neighbors(KNNGraph graph) {
	Neighbor **reverse;
	CHECK_CALL(reverse = malloc(sizeof(Neighbor*) * graph->points), NULL);
	for (size_t i = 0UL; i < graph->points; ++i)
		reverse[i] = vector_create(Neighbor, VEC_MIN_CAP);

	for (size_t i = 0UL; i < graph->points; ++i) {
		Neighbor *neighbors = graph->neighbors[i];
		for (size_t j = 0UL; j < graph->k; ++j) {
			Neighbor entry = neighbors[j]; 
			vector_insert(reverse[entry.id],
				((Neighbor){.id = i, .dist = entry.dist, .flag = entry.flag})); 
		}
	}
	return reverse;
}


Neighbor *get_knearest(KNNGraph graph, float *point) {
	Neighbor *KNearest = vector_create(Neighbor, graph->k + 1);
	for (size_t i = 0UL; i < graph->points; ++i) {
		float dist = graph->dist(&graph->data[i * (graph->dim + 1)], point, graph->dim);
		vector_sorted_insert(KNearest, ((Neighbor) {.id = i, .dist = dist}));
	}

	Neighbor *ret = malloc(graph->k * sizeof(Neighbor));
	memcpy(ret, KNearest, graph->k * sizeof(Neighbor));
	vector_destroy(KNearest);

	return ret;
}


void sample(uint32_t **vec, uint32_t size, uint32_t seed) {
	if (vector_size(*vec) <= size)
		return;

	int size__ = vector_size(*vec);
	uint32_t *sampled = vector_create(uint32_t, size__);
	uint8_t *bitset = BITSET_CREATE(vector_size(*vec));
	while (vector_size(sampled) < size) {
		uint32_t i = rand_r(&seed) % vector_size(*vec);
		while (BITSET_CHECK(bitset, i))
			i = rand_r(&seed) % vector_size(*vec);

		BITSET_SET(bitset, i);
		vector_insert(sampled, (*vec)[i]);
	}
	free(bitset);

	vector_destroy(*vec);
	*vec = sampled;
}


void collect_sets(KNNGraph graph, uint32_t **old, 
	             uint32_t **new, float sample_rate) {

	for (size_t v = 0UL; v < graph->points; ++v) {
		vector_set_size(old[v], 0);
		uint32_t true_cnt = 0;
		for (size_t i = 0UL; i < graph->k; ++i) {
			true_cnt += graph->neighbors[v][i].flag;
			if (!graph->neighbors[v][i].flag)
				vector_insert(old[v], graph->neighbors[v][i].id);
		}

		uint32_t to_sample = MIN(sample_rate * graph->k, true_cnt);
		vector_set_size(new[v], 0);
		for (size_t i = 0UL; i < to_sample; ++i) {
			uint32_t node = rand() % graph->k;
			while (!graph->neighbors[v][node].flag)
				node = rand() % graph->k;
			
			graph->neighbors[v][node].flag = false;
			vector_insert(new[v], graph->neighbors[v][node].id);
		}
	}
}


Pair *collect_pairs(uint32_t *old, uint32_t *new) {
	Pair *pairs = malloc(vector_size(new) * sizeof(Pair));

	for (size_t i = 0UL; i < vector_size(new); ++i) {
		pairs[i].id = new[i];
		pairs[i].neighbors = vector_create(Neighbor, VEC_MIN_CAP);
		for (size_t j = 0UL; j < vector_size(new); ++j) {
			uint32_t u2 = new[j];
			if (pairs[i].id >= u2)
				continue;
			
			vector_insert(pairs[i].neighbors, ((Neighbor){.id = u2}));
		}

		for (size_t j = 0UL; j < vector_size(old); ++j) {
			uint32_t u2 = old[j];
			if (pairs[i].id == u2)
				continue;

			vector_insert(pairs[i].neighbors, ((Neighbor){.id = u2}));
		}
	}
	return pairs;
}


/* Add (sampled) reverse sets to old and new */
void get_reverse(KNNGraph graph, uint32_t **old, uint32_t **new, uint32_t **old_r, uint32_t **new_r, float sample_rate) {
	for (size_t v = 0UL; v < graph->points; ++v) {	
		vector_set_size(old_r[v], 0);
		vector_set_size(new_r[v], 0);
	}

	for (size_t v = 0UL; v < graph->points; ++v) {
		for (size_t i = 0UL; i < vector_size(old[v]); ++i)
			vector_insert(old_r[old[v][i]], v);

		for (size_t i = 0UL; i < vector_size(new[v]); ++i)
			vector_insert(new_r[new[v][i]], v);
	}

	for (size_t v = 0UL; v < graph->points; ++v) {
		sample(&old_r[v], sample_rate * graph->k, (time(NULL) + v) ^ omp_get_thread_num());
		sample(&new_r[v], sample_rate * graph->k, (time(NULL) + v) ^ omp_get_thread_num());
		
		vector_append(old[v], old_r[v], vector_size(old_r[v]));
		vector_append(new[v], new_r[v], vector_size(new_r[v]));
	}
}

float dot_product(float const *f1, float const *f2, uint32_t dim) {
	float sum1 = 0.0;
	float sum2 = 0.0;
	float sum3 = 0.0;
	float sum4 = 0.0;
	
	uint32_t to_unroll = dim - dim % 4;
	for (size_t i = 0UL; i < to_unroll; i += 4) {
		sum1 += f1[i] * f2[i];
		sum2 += f1[i + 1] * f2[i + 1];
		sum3 += f1[i + 2] * f2[i + 2];
		sum4 += f1[i + 3] * f2[i + 3]; 
	}

	for (size_t i = to_unroll; i < dim; ++i)
		sum1 += f1[i] * f2[i];
	
	return sum1 + sum2 + sum3 + sum4;
}


static bool str_to_int(const char *s, uint32_t *num) {
	char *endptr;
	errno = 0;
	*num = strtol(s, &endptr, 10);

	return errno == 0 && endptr != s && *endptr == '\0';
}


static bool str_to_float(const char *s, float *num) {
	char *endptr;
	errno = 0;
	*num = strtof(s, &endptr);

	return errno == 0 && endptr != s && *endptr == '\0';
}


// static void reverse_bytes(char *bytes) {
// 	int32_t temp;
// 	memcpy(&temp, bytes, sizeof(int32_t));

// 	int32_t reversed = (temp << 8) & 0xFF00FF00;
// 	reversed |= (temp >> 8) & 0xFF00FF;

// 	reversed = (reversed << 16) | ((reversed >> 16) & 0xFFFF);
// 	memcpy(bytes, &reversed, sizeof(int32_t));
// }