#ifndef MINING_H
#define MINING_H

/*******************************************************************************

    MINING -- Autolykos parallel blockMining procedure

*******************************************************************************/

#include "definitions.h"

__constant__ ctx_t ctt[2];
__constant__ uint32_t bound_[8];

void cpyCtxSymbol(ctx_t *ctx);
void cpyBSymbol(uint8_t *bound);

// unfinalized hash of message
void InitMining(
    // context
    ctx_t * ctx,
    // message
    const uint32_t * mes,
    // message length in bytes
    const uint32_t meslen
);


__global__ void BlockMiningStep1(


    // data:  mes  
    const uint32_t * data,

    // nonce base
    const uint64_t base,

    // precalculated hashes
    const uint32_t * hashes,


    uint32_t* BHashes

);
__global__ void BlockMiningStep2(
    // data:  mes  
    const uint32_t * data,
    // nonce base
    const uint64_t base,
    // block height
    const uint32_t height,
    // precalculated hashes
    const uint32_t * hashes,
    // indices of valid solutions
    uint32_t * valid ,
    uint32_t * count,
    uint32_t*  BHashes
);
#endif // MINING_H

