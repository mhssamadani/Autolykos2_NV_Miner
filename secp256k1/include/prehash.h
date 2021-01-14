#ifndef PREHASH_H
#define PREHASH_H

#include "definitions.h"

// first iteration of hashes precalculation
__global__ void InitPrehash(
    // height
    const uint32_t  height,
    // hashes
    uint32_t * hashes
);

// precalculate hashes
int Prehash(
    const int keep,
    // 
    const uint32_t * data,
    // uncomplete hash contexts
    uctx_t * uctxs,
    // hashes
    uint32_t * hashes,
    uint32_t  height,
    uint8_t  AlgVer
);


#endif // PREHASH_H
