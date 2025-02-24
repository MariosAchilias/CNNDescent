#ifndef BITSET_H
#define BITSET_H

#define BITSET_CREATE(n) \
	(calloc((n) / 8 + ((n) % 8 != 0), sizeof(uint8_t)))

#define BITSET_CHECK(bitset, i) \
	((bitset[(i) / 8] & (((uint8_t)1) << ((i) % 8))) != 0)

#define BITSET_SET(bitset, i) \
	(bitset[(i) / 8] |= ((uint8_t)1) << ((i) % 8))

#define BITSET_UNSET(bitset, i) \
	(bitset[(i) / 8] &= ~(((uint8_t)1) << ((i) % 8)))

#endif /* BITSET_H */