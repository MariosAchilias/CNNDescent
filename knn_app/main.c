#include "utils.h"
#include "rp_tree.h"

static void usage(void) {
	fprintf(stderr, "Usage: $~ ./knn_app "
		"-file [binary_file] "
		"-k [number_of_neighbors] "
		"-endian [endianess_of_the file] "
		"-dim [data_dimension] "
		"-metric [metric_function] "
		"-precision [termination_delta] "
		"-sample [sample_rate] "
		"-threads [num_threads] "
		"-trees [num_rp_trees]\n\n"
	);
	exit(EXIT_FAILURE);
}


int main(int argc, const char **argv) {
	KnnArgs args;
	memset(&args, 0, sizeof(KnnArgs));

	if (!parse_args(argc, argv, &args))
		usage();
	
	import_dataset(&args);
	KNNGraph graph = KNNGraph_create(args.data, args.metric, 
                                     args.k, args.dim, 
                                     args.points);

	/* Check for a precomputed solution for the dataset. 
     * If it already exists we simply use it to import the proper graph
	 * otherwise we compute it and store it in the ../solutions folder.
     */
	char solution[256];
	memset(solution, 0, sizeof solution);
	
	sprintf(solution, "../solutions/k%d_%s_%s", 
        args.k, 
        (args.metric == optimized_euclidean || args.metric == euclidean_dist) 
			? "euclidean" 
			: "manhattan",
		basename(args.file)
    );

	KNNGraph ground_truth;
	if (access(solution, F_OK) == 0) {
		ground_truth = KNNGraph_import_graph(solution, args.data, args.metric);
	} else {
		ground_truth = KNNGraph_create(args.data, args.metric, 
                                       args.k, args.dim, 
                                       args.points);
		printf("No precise solution found for this configuration, "
			"computing it and exporting to %s\n", solution);
		printf("Brute-force runtime: %.2fsec\n", 
			CALC_TIME(KNNGraph_bruteforce(ground_truth)));
		KNNGraph_export_graph(ground_truth, solution);
	}

	printf("NNDescent runtime: %.2fsec\n", 
		CALC_TIME(KNNGraph_nndescent(graph, args.precision, args.sample_rate, args.n_trees)));

	printf("Recall: %.4f\n", KNNGraph_recall(graph, ground_truth));

	float scan_rate = (float)graph->similarity_comparisons / 
					 (((float)graph->points * ((float)graph->points - 1)) / 2);
	printf("NNDescent scan rate: %.4f\n", scan_rate);

	KNNGraph_destroy(graph);
	KNNGraph_destroy(ground_truth);
	
	for (size_t i = 0UL; i < args.points; ++i)
		free(args.data[i]);
	free(args.data);

	return 0;
}
