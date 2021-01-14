#ifndef MINING_H
#define MINING_H

/*******************************************************************************

    MINING -- Autolykos parallel blockMining procedure

*******************************************************************************/

#include "definitions.h"

// unfinalized hash of message
void InitMining(
    // context
    ctx_t * ctx,
    // message
    const uint32_t * mes,
    // message length in bytes
    const uint32_t meslen
);

// block mining iteration
__global__ void BlockMining(

    // boundary for puzzle
    const uint32_t * bound,

    // data:  mes  ctx
    const uint32_t * data,

    // nonce base
    const uint64_t base,

    // block height
    const uint32_t height,

    // precalculated hashes
    const uint32_t * hashes,


    // indices of valid solutions
    uint32_t * valid , 

    uint32_t * count
);
#endif // MINING_H
