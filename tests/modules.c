#include "acutest.h"
#include "vector.h"
#include "bitset.h"
#include "rp_tree.h"

static int cmp_neighbor(Neighbor a, Neighbor b) {
	return fabs(a.dist - b.dist) < 1e-9 
		? a.id - b.id 
		: a.dist > b.dist ? 1 : -1;	
}


static int cmp_entries(Neighbor a, Neighbor b) {
    return !(fabs(a.dist - b.dist) < 1e-9 && a.id == b.id);
}


void test_vector(void) {
	uint32_t N = 1000;
	Neighbor *vec = vector_create(Neighbor, N / 3);

    TEST_ASSERT(vec != NULL);
    TEST_ASSERT(vector_cap(vec) == N / 3);

	uint32_t size = 0;
    /* Insert a number of entries and check that the size 
     * is properly updated and the entry exists in the proper position
     */
	for (size_t i = 0UL; i < N; ++i) {
		Neighbor entry = { .dist = (float)i, .id = i };
		vector_insert(vec, entry);
		TEST_ASSERT(vector_size(vec) == ++size);
		TEST_ASSERT(cmp_neighbor(vec[i], entry) == 0);
	}
	TEST_ASSERT(vector_size(vec) == N);

    /* Test vector_find functionality */
	for (uint32_t i = 0; i < N; ++i) {
		Neighbor entry = { .dist = (float)i, .id = i };
		TEST_ASSERT((vector_find(vec, cmp_entries, entry)) == i);
	}

    /* Test vector_append functionality */
    Neighbor *new_entries;
    CHECK_CALL(new_entries = malloc(sizeof(Neighbor) * N), NULL);
    for (uint32_t i = 0; i < N; ++i)
        new_entries[i] = (Neighbor){ .id = (N + i), .dist = (float)(N + i) };

    size += N;
    vector_append(vec, new_entries, N);
    TEST_ASSERT(vector_size(vec) == size);

    /* Delete a random element from the vector except the last one */
	uint32_t index = rand() % (vector_size(vec) - 1);
	vector_delete(vec, index);

    /* Check the size has been updated and the entry removed */
	TEST_ASSERT(--size == vector_size(vec));
	TEST_ASSERT(vector_find(vec, cmp_entries, ((Neighbor){ .id = index, .dist = (float)index })) == -1);

    /* So when we delete the ith element from the vector we move
     * all the entries starting from index i+1 till vector_size - 1 one position to the left.
     *
     * So here is a snapshot of our vector before the deletion:
     *   0    1          index          vector_size - 1        
     * [ 0 ][ 1 ] .... [ index ]..... [ vector_size - 1 ] ... free space
     * 
     * after the deletion
     *   0    1           index       index + 1         vector_size - 2  
     * [ 0 ][ 1 ] .... [ index + 1 ][ index + 2 ] ... [ vector_size - 1 ] .. free space
     */
	for (uint32_t i = 0; i < vector_size(vec); ++i) {
		Neighbor entry = {
			.id = (i + (i >= index)),
			.dist = (float)(i + (i >= index))
		};
		TEST_ASSERT(cmp_entries(entry, vec[i]) == 0);
	}
	vector_destroy(vec);
    free(new_entries);


    /* Test vector_sorted_insert functionality */


    Neighbor to_insert[] = {
        { .id = 1,  .dist = 2.0  },
        { .id = 21, .dist = 6.0  },
        { .id = 22, .dist = 2.0  },
        { .id = 11, .dist = 12.0 },
        { .id = 1,  .dist = 2.0  }, /* deep left side duplicate */
        { .id = 13, .dist = 11.0 },
        { .id = 12, .dist = 12.0 },
        { .id = 11, .dist = 9.0  },
        { .id = 12, .dist = 12.0 } /* deep right side duplicate */
    };

    /* The expected order of the elements after the insertion */
    Neighbor inserted[] = {
        { .id = 1,  .dist = 2.0  },
        { .id = 22, .dist = 2.0  },
        { .id = 21, .dist = 6.0  },
        { .id = 11, .dist = 9.0  },
        { .id = 13, .dist = 11.0 },
        { .id = 11, .dist = 12.0 },
        { .id = 12, .dist = 12.0 }
    };

    uint32_t dup1 = 4;
    uint32_t dup2 = 8;

    vec = vector_create(Neighbor, sizeof(to_insert));
    for (size_t i = 0UL; i < ARRAY_SIZE(to_insert); ++i)
        TEST_ASSERT(!(i != dup1 && i != dup2) ^ vector_sorted_insert(vec, to_insert[i]));
        
    TEST_ASSERT(vector_size(vec) == ARRAY_SIZE(inserted));
    for (size_t i = 0UL; i < vector_size(vec); ++i)
        TEST_ASSERT(cmp_entries(vec[i], inserted[i]) == 0);

    vector_destroy(vec);
}


void test_bitset(void) {

    /* Create a bitset and check:
     * 1) It is properly initialized aka all its bits are unset
     * 2) Set a number of bits and check that are indeed set.
     * 3) Count the bits that have been set in the bitset and check 
     *    that it matches the number from step 2
     * 4) Unset every bit in the bitset
     * 5) Count the bits that have been set. We expect the counter to be zero 
     */
    uint32_t N = 10000;

    uint8_t *bitset;
    CHECK_CALL(bitset = BITSET_CREATE(N), NULL);

    TEST_ASSERT(bitset != NULL);
    for (uint32_t i = 0; i < N; ++i)
        TEST_ASSERT(!BITSET_CHECK(bitset, i));

    for (uint32_t i = 0; i < N; i += 2) {
        BITSET_SET(bitset, i);
        TEST_ASSERT(BITSET_CHECK(bitset, i));
        TEST_ASSERT(!BITSET_CHECK(bitset, i + 1));
    }

    uint32_t set_count = 0;
    for (uint32_t i = 0; i < N; i++)
        set_count += BITSET_CHECK(bitset, i);    
    TEST_ASSERT(set_count == N / 2);

    for (uint32_t i = 0; i < N; i += 2) {
        BITSET_UNSET(bitset, i);
        TEST_ASSERT(!BITSET_CHECK(bitset, i));
        set_count--;
    }
    
    for (uint32_t i = 0; i < N; ++i)
        set_count += BITSET_CHECK(bitset, i);
    
    TEST_ASSERT(set_count == 0);
    free(bitset);
}

float *test_data[5] = {	
	(float[]){ 1.0, 1.0, 1.0, 1.0},
	(float[]){ 2.0, 2.0, 2.0, 2.0},
	(float[]){ 3.0, 3.0, 3.0, 3.0},
	(float[]){ 4.0, 4.0, 4.0, 4.0},
	(float[]){ 5.0, 5.0, 5.0, 5.0}
};


void test_rptree(void) {
    uint32_t leaf_size = 4;
    uint32_t N = 5;
    uint32_t dim = ARRAY_SIZE(test_data) - 1;
    RPTree tree = RPTree_create(test_data, N, dim, leaf_size);

    TEST_ASSERT(tree != NULL);
    TEST_ASSERT(tree->nodes != NULL);
    TEST_ASSERT(tree->leaf_size == leaf_size);
    TEST_ASSERT(tree->dimension == dim);
    TEST_ASSERT(tree->points == N);


    RPTree_split(tree);
    uint32_t points_found = 0;
    for (size_t i = 0UL; i < vector_size(tree->nodes); i++) {
        /* If leaf node */
        if (tree->nodes[i].data != NULL) {
            TEST_ASSERT(vector_size(tree->nodes[i].data) <= leaf_size);
            points_found += vector_size(tree->nodes[i].data);
            /* Leaf nodes store no split information */
            TEST_ASSERT(tree->nodes[i].hyperplane == NULL);
        }
        /* If internal node */
        if (tree->nodes[i].hyperplane != NULL)
            /* Internal nodes store no data */
            TEST_ASSERT(tree->nodes[i].data == NULL);
    }
    TEST_ASSERT(points_found == N);

    RPTree_destroy(tree);
}

TEST_LIST = {
    { "test_vector", test_vector },
    { "test_bitset", test_bitset },
    { "test_rptree", test_rptree },

    { NULL, NULL }
};
