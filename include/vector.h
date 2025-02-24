#ifndef VECTOR_H
#define VECTOR_H

#include "common.h"
#define VEC_MIN_CAP 16


struct vector {
	omp_lock_t lock;
	uint32_t size;
	uint32_t cap;
};


#define vector_factor(__size, __cap) ({		    \
	uint32_t __mult = 1;					    \
	while ((__size) > ((__cap) << __mult++))	\
		;									    \
	((__cap) << __mult);						\
})


#define vector_set_size(vec, __size) {					\
	((&((struct vector*)(vec))[-1])->size = (__size));	\
}


#define vector_set_cap(vec, __cap) {					\
	((&((struct vector*)(vec))[-1])->cap = (__cap));	\
}

#define vector_lock(vec) ({						\
	&((&((struct vector*)(vec))[-1])->lock);	\
})


#define vector_size(vec) ({					\
	((&((struct vector*)(vec))[-1])->size);	\
})


#define vector_cap(vec) ({					\
	((&((struct vector*)(vec))[-1])->cap);	\
})


#define vector_destroy(vec) {				\
	free((&((struct vector*)(vec))[-1]));	\
}


#define vector_create(type, __cap) ({										\
	uint64_t bytes = sizeof(type) * ((__cap) + 1) + sizeof(struct vector);	\
	char *vec = (char*)calloc(bytes, 1) + sizeof(struct vector);			\
	vector_set_size(vec, 0);												\
	vector_set_cap(vec, __cap);												\
	((type*)vec);															\
})


#define vector_grow(vec, __new_cap) {												\
	uint64_t new_size = sizeof(*vec) * ((__new_cap) + 1) + sizeof(struct vector);	\
	vector_set_cap(vec, __new_cap);													\
	void *__vec = (&((struct vector*)(vec))[-1]);									\
	vec = (void*)((char*)realloc(__vec, new_size) + sizeof(struct vector));			\
}


#define vector_insert(vec, __entry) {	\
	uint32_t __size = vector_size(vec);	\
	uint32_t __cap  = vector_cap(vec);	\
	if (__size == __cap)				\
		vector_grow(vec, (__cap << 1));	\
	(vec)[__size] = (__entry);			\
	vector_set_size(vec, (__size + 1));	\
}

#define vector_sorted_insert(vec, __entry) ({							\
	__label__ _ret;														\
	int low = 0;														\
	int high = vector_size(vec) - 1;									\
	bool inserted = false;												\
																		\
	if (high > 0 && vector_size(vec) == vector_cap(vec)					\
	 && vec[high].dist < __entry.dist)									\
	 	goto _ret;														\
																		\
	while (low <= high) {												\
		int mid = low + (high - low) / 2;								\
		int mid_ = mid;													\
		while (mid_ >= 0 && vec[mid_].dist == __entry.dist)				\
			if (vec[mid_--].id == __entry.id)							\
				goto _ret;												\
																		\
		if (vec[mid].dist > __entry.dist)								\
			high = mid - 1;												\
		else															\
			low = mid + 1;												\
	}																	\
	if (vec[low].dist > __entry.dist)									\
		memmove(&vec[low + 1], 											\
			    &vec[low],												\
			    (vector_size(vec) - low) * sizeof(*vec));				\
																		\
	vec[low] = __entry;													\
																		\
	if (vector_size(vec) != vector_cap(vec))							\
		vector_set_size(vec, vector_size(vec) + 1);						\
																		\
	inserted = true;													\
	_ret:																\
		inserted;														\
})


#define vector_delete(vec, __index) {								\
	if (0 <= (__index) && (__index) <= (vector_size(vec) - 1)) {	\
		memmove(													\
			&vec[__index],											\
			&vec[__index] + 1UL,									\
			(vector_size(vec) - __index - 1UL) * sizeof(*vec)		\
		);															\
		vector_set_size(vec, (vector_size(vec) - 1));				\
	}																\
}


#define vector_find(vec, __cmp, __entry) ({						\
	size_t __index = 0UL;										\
	while (__index < vector_size(vec)							\
		&& __cmp(vec[__index++], __entry) != 0)					\
		;														\
	__cmp(vec[__index - 1], __entry) == 0 ? __index - 1 : -1;	\
})


#define vector_append(vec, __array, __size) {								\
	uint32_t space_left = vector_cap(vec) - vector_size(vec);				\
	if (space_left < (__size))												\
		vector_grow(vec, vector_factor(__size, vector_cap(vec)));			\
	memcpy(vec + (vector_size(vec)), (__array), (__size) * sizeof(*vec));	\
	vector_set_size(vec, (vector_size(vec) + __size));						\
}


#endif /* VECTOR_H */