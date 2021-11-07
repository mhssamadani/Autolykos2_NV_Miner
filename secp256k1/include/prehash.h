#ifndef PREHASH_H
#define PREHASH_H

#include "definitions.h"

// first iteration of hashes precalculation
__global__ void InitPrehash(
    const uint32_t  n_len,
    // height
    const uint32_t height,
    // hashes
    uint32_t *hashes);

// precalculate hashes
int Prehash(
    uint64_t N_LEN,
    // hashes
    uint32_t *hashes,
    uint32_t height);

#endif // PREHASH_H

