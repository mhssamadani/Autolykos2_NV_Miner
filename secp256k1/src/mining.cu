// mining.cu

/*******************************************************************************

    MINING -- Autolykos parallel BlockMining procedure

*******************************************************************************/

#include "../include/mining.h"
#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////
//  Unfinalized hash of message
////////////////////////////////////////////////////////////////////////////////
void InitMining(
    // context
    ctx_t * ctx,
    // message
    const uint32_t * mes,
    // message length in bytes
    const uint32_t meslen
)
{
    uint64_t aux[32];

    //========================================================================//
    //  Initialize context
    //========================================================================//
    memset(ctx->b, 0, BUF_SIZE_8);
    B2B_IV(ctx->h);
    ctx->h[0] ^= 0x01010000 ^ NUM_SIZE_8;
    memset(ctx->t, 0, 16);
    ctx->c = 0;

    //========================================================================//
    //  Hash message
    //========================================================================//
    for (uint_t j = 0; j < meslen; ++j)
    {
        if (ctx->c == BUF_SIZE_8) { HOST_B2B_H(ctx, aux); }

        ctx->b[ctx->c++] = ((const uint8_t *)mes)[j];
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Block mining                                                               
////////////////////////////////////////////////////////////////////////////////
static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, 0, 0x0123);
}
__global__	void BlockMiningPH1(
    const uint32_t * ctx_m,
    // nonce base
    const uint64_t base,
    ctx_t * ctx_mn
    )
 {
        uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
		uint32_t htid = tid * 256;
        uint64_t  aux[32];
        ctx_t sdata;
        #pragma unroll
            for (int i = 0; i < CTX_SIZE; ++i)
            {
                ((uint8_t * )&sdata)[i] = (( uint8_t * )ctx_m)[i];
            }
            ctx_t *ctx = ((ctx_t * )(&sdata));
            if (tid < NONCES_PER_ITER)
            {
                uint32_t non[NONCE_SIZE_32];
        
                asm volatile (
                    "add.cc.u32 %0, %1, %2;":
                    "=r"(non[0]): "r"(((uint32_t *)&base)[0]), "r"(htid)
                );
        
                asm volatile (
                    "addc.u32 %0, %1, 0;": "=r"(non[1]): "r"(((uint32_t *)&base)[1])
                );
        

			//================================================================//
			//  Hash 7 byte nonce
			//================================================================//
            #pragma unroll
			for (uint_t j = 0; j < NONCE_SIZE_8-1; ++j)
			{
				if (ctx->c == BUF_SIZE_8) { DEVICE_B2B_H(ctx, aux); }

				ctx->b[ctx->c++] = ((uint8_t *)non)[NONCE_SIZE_8 - j - 1];
			}
			

			//================================================================//
			//  Load to global memory
			//================================================================//
            #pragma unroll
			for (int i = 0; i < CTX_SIZE; ++i)
			{
				((uint8_t * )&ctx_mn[tid])[i] = ((uint8_t * )&sdata)[i]; 

			}
        
    }    
        
}


__global__ void BlockMiningPH2(

    // boundary for puzzle
    const uint32_t * bound,

    // data:  mes  
    const uint32_t * mes,

    ctx_t * ctx_mn, 
    // nonce base
    const uint64_t base,

    // block height
    const uint32_t height,

    // precalculated hashes
    const uint32_t * hashes,


    // indices of valid solutions
    uint32_t * valid , 

    uint32_t * count
)
{
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    ctx_t sdata;


    // 472 bytes
    uint32_t ldata[118];

    // 256 bytes
    uint64_t * aux = (uint64_t *)ldata;
    // (4 * K_LEN) bytes
    uint32_t * ind = ldata;
    // (NUM_SIZE_8 + 4) bytes
    uint32_t * r = ind + K_LEN;

    

    if (tid < NONCES_PER_ITER)
    {
        uint32_t j;
        uint32_t non[NONCE_SIZE_32];

        asm volatile (
            "add.cc.u32 %0, %1, %2;":
            "=r"(non[0]): "r"(((uint32_t *)&base)[0]), "r"(tid)
        );

        asm volatile (
            "addc.u32 %0, %1, 0;": "=r"(non[1]): "r"(((uint32_t *)&base)[1])
        );

        uint32_t ctxIndx = tid / 256 ;
        for (int i = 0; i < CTX_SIZE; ++i)
        {
            ((uint8_t * )&sdata)[i] = ((uint8_t * )&ctx_mn[ctxIndx])[i];

        }
        ctx_t *ctx = ((ctx_t * )(&sdata));

        //================================================================//
        //  Hash last byte of nonce
        //================================================================//

        if (ctx->c == 128) { HOST_B2B_H(ctx, aux); }
        ctx->b[ctx->c++] = ((uint8_t *)non)[0];
        //================================================================//
        //  Finalize hashes
        //================================================================//
        DEVICE_B2B_H_LAST(ctx, aux);

#pragma unroll
        for (j = 0; j < NUM_SIZE_8; ++j)
        {
            ((uint8_t *)r)[j]
                = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
        }
        uint64_t h2;
        ((uint8_t*)&h2)[0] = ((uint8_t*)r)[31];
        ((uint8_t*)&h2)[1] = ((uint8_t*)r)[30];
        ((uint8_t*)&h2)[2] = ((uint8_t*)r)[29];
        ((uint8_t*)&h2)[3] = ((uint8_t*)r)[28];
        ((uint8_t*)&h2)[4] = ((uint8_t*)r)[27];
        ((uint8_t*)&h2)[5] = ((uint8_t*)r)[26];
        ((uint8_t*)&h2)[6] = ((uint8_t*)r)[25];
        ((uint8_t*)&h2)[7] = ((uint8_t*)r)[24];

        uint32_t h3 = h2 % N_LEN;

        for(int i=0;i<8;++i)
        {
            r[7-i]  = cuda_swab32( hashes[(h3 << 3) + i ] );
        }
/*
	        if(tid == 0)
        {
            for (int i = 0; i < NUM_SIZE_8; ++i) 
            {
                printf("\n %d %d ",  i , ((uint8_t *)r)[i]);
            }    
        }
*/
        //====================================================================//
        //  Initialize context
        //====================================================================//
        #pragma unroll
        for (int am = 0; am < BUF_SIZE_8; am++)
        {
            ctx->b[am] = 0;
        }
        B2B_IV(ctx->h);



        ctx->h[0] ^= 0x01010000 ^ NUM_SIZE_8;
        //memset(ctx->t, 0, 16);
        ctx->t[0] = 0;
        ctx->t[1] = 0;
        ctx->c = 0;

        //====================================================================//
        //  Hash 
        //====================================================================//
#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < NUM_SIZE_8 - 1; ++j)
        {
            ctx->b[ctx->c++] = ((const uint8_t *)r)[j + 1];
        }

        //====================================================================//
        //  Hash message
        //====================================================================//
#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < NUM_SIZE_8; ++j)
        {
            ctx->b[ctx->c++] = (( const uint8_t *)mes)[j];
        }



        while (j < NUM_SIZE_8)
        {
            HOST_B2B_H(ctx, aux);

            while (ctx->c < BUF_SIZE_8 && j < NUM_SIZE_8)
            {
                ctx->b[ctx->c++] = (( const uint8_t *)mes)[j++];
            }
        }

        //================================================================//
        //  Hash nonce
        //================================================================//

#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < NONCE_SIZE_8; ++j)
        {
            ctx->b[ctx->c++] = ((uint8_t *)non)[NONCE_SIZE_8 - j - 1];
        }

#pragma unroll
        for (; j < NONCE_SIZE_8;)
        {
            HOST_B2B_H(ctx, aux);

#pragma unroll
            for (; ctx->c < BUF_SIZE_8 && j < NONCE_SIZE_8; ++j)
            {
                ctx->b[ctx->c++] = ((uint8_t*)non)[NONCE_SIZE_8 - j - 1];
            }
        }
        //====================================================================//
        //  Finalize hash
        //====================================================================//



        HOST_B2B_H_LAST(ctx, aux);


#pragma unroll
        for (j = 0; j < NUM_SIZE_8; ++j)
        {
            ((uint8_t*)r)[(j & 0xFFFFFFFC) + (3 - (j & 3))] = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
        }

        //================================================================//
        //  Generate indices
        //================================================================//
#pragma unroll
        for (int i = 1; i < INDEX_SIZE_8; ++i)
        {
            ((uint8_t *)r)[NUM_SIZE_8 + i] = ((uint8_t *)r)[i];
        }

#pragma unroll
        for (int k = 0; k < K_LEN; k += INDEX_SIZE_8) 
        { 
            ind[k] = r[k >> 2] & N_MASK; 
        
#pragma unroll 
            for (int i = 1; i < INDEX_SIZE_8; ++i) 
            { 
                ind[k + i] 
                    = (
                        (r[k >> 2] << (i << 3))
                        | (r[(k >> 2) + 1] >> (32 - (i << 3)))
                    ) & N_MASK; 
            } 
        } 


        //================================================================//
        //  Calculate result
        //================================================================//
 
        // first addition of hashes -> r
        asm volatile (
            "add.cc.u32 %0, %1, %2;":
            "=r"(r[0]): "r"(hashes[ind[0] << 3]), "r"(hashes[ind[1] << 3])
        );

#pragma unroll
        for (int i = 1; i < 8; ++i)
        {
            asm volatile (
                "addc.cc.u32 %0, %1, %2;":
                "=r"(r[i]):
                "r"(hashes[(ind[0] << 3) + i]),
                "r"(hashes[(ind[1] << 3) + i])
            );
        }

        asm volatile ("addc.u32 %0, 0, 0;": "=r"(r[8]));

        // remaining additions
#pragma unroll
        for (int k = 2; k < K_LEN; ++k)
        {
            asm volatile (
                "add.cc.u32 %0, %0, %1;":
                "+r"(r[0]): "r"(hashes[ind[k] << 3])
            );

#pragma unroll
            for (int i = 1; i < 8; ++i)
            {
                asm volatile (
                    "addc.cc.u32 %0, %0, %1;":
                    "+r"(r[i]): "r"(hashes[(ind[k] << 3) + i])
                );
            }

            asm volatile ("addc.u32 %0, %0, 0;": "+r"(r[8]));
        }


        

        //--------------------hash(f)--------------------
        //====================================================================//
        //  Initialize context
        //====================================================================//
        //memset(ctx->b, 0, BUF_SIZE_8);
#pragma unroll
        for (int am = 0; am < BUF_SIZE_8; am++)
        {
            ctx->b[am] = 0;
        }
        B2B_IV(ctx->h);



        ctx->h[0] ^= 0x01010000 ^ NUM_SIZE_8;
        //memset(ctx->t, 0, 16);
        ctx->t[0] = 0;
        ctx->t[1] = 0;
        ctx->c = 0;


        //--------------hash--------------------
#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < NUM_SIZE_8; ++j)
        {
            ctx->b[ctx->c++] = ((const uint8_t *)r)[NUM_SIZE_8 - j - 1];
        }

        //====================================================================//
        //  Finalize hash
        //====================================================================//


        HOST_B2B_H_LAST(ctx, aux);


#pragma unroll
        for (j = 0; j < NUM_SIZE_8; ++j)
        {
            ((uint8_t*)r)[NUM_SIZE_8 - j - 1] = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
        }



        //================================================================//
        //  Dump result to global memory -- LITTLE ENDIAN
        //================================================================//
        j = ((uint64_t *)r)[3] < ((uint64_t *)bound)[3]
            || ((uint64_t *)r)[3] == ((uint64_t *)bound)[3] && (
                ((uint64_t *)r)[2] < ((uint64_t *)bound)[2]
                || ((uint64_t *)r)[2] == ((uint64_t *)bound)[2] && (
                    ((uint64_t *)r)[1] < ((uint64_t *)bound)[1]
                    || ((uint64_t *)r)[1] == ((uint64_t *)bound)[1]
                    && ((uint64_t *)r)[0] < ((uint64_t *)bound)[0]
                )
            );

        
            if(j )
            {
    
                
                uint32_t id = atomicInc(count, MAX_SOLS);
                valid[id] = tid+1; 
           }
        }

    return;

}

// mining.cu



