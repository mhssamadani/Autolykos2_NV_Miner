// prehash.cu

/*******************************************************************************

    PREHASH -- precalculation of hashes

*******************************************************************************/

#include "../include/prehash.h"
#include "../include/compaction.h"
#include "../include/definitions.h"
#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////
//  Precalculate hashes
////////////////////////////////////////////////////////////////////////////////
int Prehash(
    const int keep,
    // data: pk || mes || w || padding || x || sk
    const uint32_t * data,
    // unfinalized hashes contexts
    uctx_t * uctxs,
    // hashes
    uint32_t * hashes,
    uint32_t  height , 
    uint8_t  AlgVer 
)
{

    if(AlgVer == 1)
    {
        return EXIT_FAILURE ;
    }
    else
    {
        InitPrehash<<<1 + (N_LEN - 1) / BLOCK_DIM, BLOCK_DIM>>>(
            height, hashes
        );
        CUDA_CALL(cudaPeekAtLastError());

    }
    return EXIT_SUCCESS;
    
}
__global__ void InitPrehash(
    // height
    const uint32_t  height,
    // hashes
    uint32_t * hashes
)
{
    uint32_t tid = threadIdx.x;

    // shared memory
    __shared__ uint32_t sdata[ROUND_PNP_SIZE_32];

    tid += blockDim.x * blockIdx.x;

    if (tid < N_LEN)
    {
        uint32_t j;

        // local memory
        // 472 bytes
        uint32_t ldata[118];

        // 32 * 64 bits = 256 bytes 
        uint64_t * aux = (uint64_t *)ldata;
        // (212 + 4) bytes 
        ctx_t * ctx = (ctx_t *)(ldata + 64);

        //====================================================================//
        //  Initialize context
        //====================================================================//
        memset(ctx->b, 0, BUF_SIZE_8);
        B2B_IV(ctx->h);
        ctx->h[0] ^= 0x01010000 ^ NUM_SIZE_8;
        memset(ctx->t, 0, 16);
        ctx->c = 0;

        //====================================================================//
        //  Hash tid
        //====================================================================//
#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < INDEX_SIZE_8; ++j)
        {
            ctx->b[ctx->c++] = ((const uint8_t *)&tid)[INDEX_SIZE_8 - j - 1];
        }

        //====================================================================//
        //  Hash height
        //====================================================================//
        #pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < HEIGHT_SIZE ; ++j)
        {
            ctx->b[ctx->c++] = ((const uint8_t *)&height)[j];
        }

        //====================================================================//
        //  Hash constant message
        //====================================================================//
#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < CONST_MES_SIZE_8; ++j)
        {
            ctx->b[ctx->c++]
                = (
                    !((7 - (j & 7)) >> 1)
                    * ((j >> 3) >> (((~(j & 7)) & 1) << 3))
                ) & 0xFF;
        }

        while (j < CONST_MES_SIZE_8)
        {
            DEVICE_B2B_H(ctx, aux);

            for ( ; ctx->c < BUF_SIZE_8 && j < CONST_MES_SIZE_8; ++j)
            {
                ctx->b[ctx->c++]
                    = (
                        !((7 - (j & 7)) >> 1)
                        * ((j >> 3) >> (((~(j & 7)) & 1) << 3))
                    ) & 0xFF;
            }
        }

         //====================================================================//
        //  Finalize hash
        //====================================================================//
        DEVICE_B2B_H_LAST(ctx, aux);

#pragma unroll
        for (j = 0; j < NUM_SIZE_8; ++j)
        {
            ((uint8_t *)ldata)[NUM_SIZE_8 - j - 1]
                = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
        }

        //====================================================================//
        //  Dump result to global memory -- BIG ENDIAN
        //====================================================================//
#pragma unroll
        for (int i = 0; i < NUM_SIZE_8-1; ++i) 
        {
            ((uint8_t *)hashes)[tid * NUM_SIZE_8 +i ]
                = ((uint8_t *)ldata)[i];
        }
        ((uint8_t *)hashes)[tid * NUM_SIZE_8 +NUM_SIZE_8-1 ] = 0; 
 
    }

    return;
}
// prehash.cu




