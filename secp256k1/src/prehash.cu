// prehash.cu

/*******************************************************************************

    PREHASH -- precalculation of hashes

*******************************************************************************/

#include "../include/prehash.h"
#include "../include/compaction.h"
#include "../include/definitions.h"
#include <cuda.h>

__device__ __forceinline__
static void store32(ctx_t *ctx, uint32_t num32)
{
	int len = ctx->c;
	ctx->b[0 + len] = (num32 >> 24) & 0xFF;
	ctx->b[1 + len] = (num32 >> 16) & 0xFF;
	ctx->b[2 + len] = (num32 >> 8) & 0xFF;
	ctx->b[3 + len] = num32 & 0xFF;
	ctx->c += 4;
}

__device__ __forceinline__
static void store64(ctx_t *ctx, uint64_t num64)
{
	int len = ctx->c;
	ctx->b[0 + len] = (num64 >> 56) & 0xFF;
	ctx->b[1 + len] = (num64 >> 48) & 0xFF;
	ctx->b[2 + len] = (num64 >> 40) & 0xFF;
	ctx->b[3 + len] = (num64 >> 32) & 0xFF;
	ctx->b[4 + len] = (num64 >> 24) & 0xFF;
	ctx->b[5 + len] = (num64 >> 16) & 0xFF;
	ctx->b[6 + len] = (num64 >> 8) & 0xFF;
	ctx->b[7 + len] = num64 & 0xFF;
	ctx->c += 8;
}

////////////////////////////////////////////////////////////////////////////////
//  Precalculate hashes
////////////////////////////////////////////////////////////////////////////////
int Prehash(
    uint32_t * hashes,
    uint32_t  height 
)
{

    InitPrehash<<<1 + (N_LEN - 1) / BLOCK_DIM, BLOCK_DIM>>>(
        height, hashes
    );
    CUDA_CALL(cudaPeekAtLastError());

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
		store32(ctx,  tid);

        //====================================================================//
        //  Hash height
        //====================================================================//
		//store32(ctx,  height);
        #pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < HEIGHT_SIZE ; ++j)
        {
            ctx->b[ctx->c++] = ((const uint8_t *)&height)[j];
        }

        //====================================================================//
        //  Hash constant message
        //====================================================================//
		uint64_t k=0;
		#pragma unroll 15
		for (int i=0; i<15; i++)
        {
			store64(ctx, k);
			k++;
		}
		//store64_hi(ctx, k);
		DEVICE_B2B_H(ctx, aux); //128-bytes

		#pragma unroll 63
		for (int j=0; j<63; j++)
        {
			//store64_low(ctx, k);
			#pragma unroll 16
			for (int i=0; i<16; i++)
			{
				store64(ctx, k);
				k++;
			}
			//k++;
			//store64_hi(ctx, k);
			DEVICE_B2B_H(ctx, aux);
		}
		//store64_low(ctx, k);

		store64(ctx, k);

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




