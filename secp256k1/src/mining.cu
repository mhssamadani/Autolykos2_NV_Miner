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
__global__ void BlockMining(
    // boundary for puzzle
    const uint32_t * bound,
    // data: pk || mes || w || padding || x || sk || ctx
    const uint32_t * data,
    // nonce base
    const uint64_t base,
    // precalculated hashes
    const uint32_t * hashes,
    // results
    uint32_t * res,
    // indices of valid solutions
    uint32_t * valid ,
 	uint32_t *count
)
{
    uint32_t tid = threadIdx.x;

    // shared memory
    __shared__ uint32_t sdata[ROUND_NC_SIZE_32];

#pragma unroll
    for (int i = 0; i < NC_SIZE_32_BLOCK; ++i)
    {
        sdata[NC_SIZE_32_BLOCK * tid + i]
            = data[
                NC_SIZE_32_BLOCK * tid + NUM_SIZE_32 * 2
                + COUPLED_PK_SIZE_32 + i
            ];
    }

    __syncthreads();

    // NUM_SIZE_8 bytes
    uint32_t * sk = sdata;

    // local memory
    // 472 bytes
    uint32_t ldata[118];

    // 256 bytes
    uint64_t * aux = (uint64_t *)ldata;
    // (4 * K_LEN) bytes
    uint32_t * ind = ldata;
    // (NUM_SIZE_8 + 4) bytes
    uint32_t * r = ind + K_LEN;
    // (212 + 4) bytes 
    ctx_t * ctx = (ctx_t *)(ldata + 64);

#pragma unroll
    for (int t = 0; t < NONCES_PER_THREAD; ++t) 
    {
        *ctx = *((ctx_t *)(sdata + NUM_SIZE_32));

        tid = threadIdx.x + blockDim.x * blockIdx.x
            + t * gridDim.x * blockDim.x;

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

            //================================================================//
            //  Hash nonce
            //================================================================//
#pragma unroll
            for (j = 0; ctx->c < BUF_SIZE_8 && j < NONCE_SIZE_8; ++j)
            {
                ctx->b[ctx->c++] = ((uint8_t *)non)[NONCE_SIZE_8 - j - 1];
            }

#pragma unroll
            for ( ; j < NONCE_SIZE_8; )
            {
                DEVICE_B2B_H(ctx, aux);
               
#pragma unroll
                for ( ; ctx->c < BUF_SIZE_8 && j < NONCE_SIZE_8; ++j)
                {
                    ctx->b[ctx->c++] = ((uint8_t *)non)[NONCE_SIZE_8 - j - 1];
                }
            }

            //================================================================//
            //  Finalize hashes
            //================================================================//
            DEVICE_B2B_H_LAST(ctx, aux);

#pragma unroll
            for (j = 0; j < NUM_SIZE_8; ++j)
            {
                ((uint8_t *)r)[(j & 0xFFFFFFFC) + (3 - (j & 3))]
                    = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
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

            // subtraction of secret key
            asm volatile ("sub.cc.u32 %0, %0, %1;": "+r"(r[0]): "r"(sk[0]));

#pragma unroll
            for (int i = 1; i < 8; ++i)
            {
                asm volatile (
                    "subc.cc.u32 %0, %0, %1;": "+r"(r[i]): "r"(sk[i])
                );
            }

            asm volatile ("subc.u32 %0, %0, 0;": "+r"(r[8]));

            //================================================================//
            //  Result mod Q
            //================================================================//
            // 20 bytes
            uint32_t * med = ind;
            // 4 bytes
            uint32_t * d = ind + 5; 
            uint32_t * carry = d;

            d[0] = r[8];

            //================================================================//
            asm volatile (
                "mul.lo.u32 %0, %1, " q0_s ";": "=r"(med[0]): "r"(*d)
            );

            asm volatile (
                "mul.hi.u32 %0, %1, " q0_s ";": "=r"(med[1]): "r"(*d)
            );

            asm volatile (
                "mul.lo.u32 %0, %1, " q2_s ";": "=r"(med[2]): "r"(*d)
            );

            asm volatile (
                "mul.hi.u32 %0, %1, " q2_s ";": "=r"(med[3]): "r"(*d)
            );

            asm volatile (
                "mad.lo.cc.u32 %0, %1, " q1_s ", %0;": "+r"(med[1]): "r"(*d)
            );

            asm volatile (
                "madc.hi.cc.u32 %0, %1, " q1_s ", %0;": "+r"(med[2]): "r"(*d)
            );

            asm volatile (
                "madc.lo.cc.u32 %0, %1, " q3_s ", %0;": "+r"(med[3]): "r"(*d)
            );

            asm volatile (
                "madc.hi.u32 %0, %1, " q3_s ", 0;": "=r"(med[4]): "r"(*d)
            );

            //================================================================//
            asm volatile ("sub.cc.u32 %0, %0, %1;": "+r"(r[0]): "r"(med[0]));

#pragma unroll
            for (int i = 1; i < 5; ++i)
            {
                asm volatile (
                    "subc.cc.u32 %0, %0, %1;": "+r"(r[i]): "r"(med[i])
                );
            }

#pragma unroll
            for (int i = 5; i < 7; ++i)
            {
                asm volatile ("subc.cc.u32 %0, %0, 0;": "+r"(r[i]));
            }

            asm volatile ("subc.u32 %0, %0, 0;": "+r"(r[7]));

            //================================================================//
            d[1] = d[0] >> 31;
            d[0] <<= 1;

            asm volatile ("add.cc.u32 %0, %0, %1;": "+r"(r[4]): "r"(d[0]));
            asm volatile ("addc.cc.u32 %0, %0, %1;": "+r"(r[5]): "r"(d[1]));
            asm volatile ("addc.cc.u32 %0, %0, 0;": "+r"(r[6]));
            asm volatile ("addc.u32 %0, %0, 0;": "+r"(r[7]));

            //================================================================//
            asm volatile ("sub.cc.u32 %0, %0, " q0_s ";": "+r"(r[0]));
            asm volatile ("subc.cc.u32 %0, %0, " q1_s ";": "+r"(r[1]));
            asm volatile ("subc.cc.u32 %0, %0, " q2_s ";": "+r"(r[2]));
            asm volatile ("subc.cc.u32 %0, %0, " q3_s ";": "+r"(r[3]));
            asm volatile ("subc.cc.u32 %0, %0, " q4_s ";": "+r"(r[4]));

#pragma unroll
            for (int i = 5; i < 8; ++i)
            {
                asm volatile ("subc.cc.u32 %0, %0, " qhi_s ";": "+r"(r[i]));
            }

            asm volatile ("subc.u32 %0, 0, 0;": "=r"(*carry));

            *carry = 0 - *carry;

            //================================================================//
            asm volatile (
                "mad.lo.cc.u32 %0, %1, " q0_s ", %0;": "+r"(r[0]): "r"(*carry)
            );

            asm volatile (
                "madc.lo.cc.u32 %0, %1, " q1_s ", %0;": "+r"(r[1]): "r"(*carry)
            );

            asm volatile (
                "madc.lo.cc.u32 %0, %1, " q2_s ", %0;": "+r"(r[2]): "r"(*carry)
            );

            asm volatile (
                "madc.lo.cc.u32 %0, %1, " q3_s ", %0;": "+r"(r[3]): "r"(*carry)
            );

            asm volatile (
                "madc.lo.cc.u32 %0, %1, " q4_s ", %0;": "+r"(r[4]): "r"(*carry)
            );

#pragma unroll
            for (int i = 5; i < 7; ++i)
            {
                asm volatile (
                    "madc.lo.cc.u32 %0, %1, " qhi_s ", %0;":
                    "+r"(r[i]): "r"(*carry)
                );
            }

            asm volatile (
                "madc.lo.u32 %0, %1, " qhi_s ", %0;": "+r"(r[7]): "r"(*carry)
            );

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

            

            if(j)
            {
    
                
                    uint32_t id = atomicInc(count, MAX_SOLS);
                    valid[id] = tid+1; 
                    #pragma unroll
                    for (int i = 0; i < NUM_SIZE_32; ++i)
                    {
                        res[i + id*NUM_SIZE_32] = r[i];
                    }
            }
        }

        __syncthreads();
    }

    return;
}


// block mining iteration
__global__ void BlockMiningV22(
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
    // results
    uint32_t * res,
    // indices of valid solutions
    uint32_t * valid , 
    uint32_t * count
)
{
    uint32_t tid = threadIdx.x;

    ctx_t sdata;
#pragma unroll
	for (int i = 0; i < CTX_SIZE; ++i)
	{
		((uint8_t * )&sdata)[i] = (( uint8_t * )data)[NUM_SIZE_8 + i];

	}
	ctx_t *ctx = ((ctx_t * )(&sdata));//&lctx;//(ctx_t *)(ldata + 64);

   // __syncthreads();

    // NUM_SIZE_8 bytes
    //uint32_t * sk = sdata;

    // local memory
    // 472 bytes
    uint32_t ldata[118];

    // 256 bytes
    uint64_t * aux = (uint64_t *)ldata;
    // (4 * K_LEN) bytes
    uint32_t * ind = ldata;
    // (NUM_SIZE_8 + 4) bytes
    uint32_t * r = ind + K_LEN;
	uint32_t pre_r[32];  /*????*/

        
        tid = threadIdx.x + blockDim.x * blockIdx.x;

//        if(tid != 0)
//            return ;

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

  //          printf( "\n base: %d  ",base);
    //        printf( "\n H: %d  ",height);

      //      printf( " \n -----------1------------- ");
        //    printf("\n cytx->c : %d " , ctx->c);
          //  for (int k = 0; k < ctx->c; ++k)
           // {
             //   printf( "\n %d  ",ctx->b[k]);
          ///  }
    
            //================================================================//
            //  Hash nonce
            //================================================================//
#pragma unroll
            for (j = 0; ctx->c < BUF_SIZE_8 && j < NONCE_SIZE_8; ++j)
            {
                ctx->b[ctx->c++] = ((uint8_t *)non)[NONCE_SIZE_8 - j - 1];
            }

#pragma unroll
            for ( ; j < NONCE_SIZE_8; )
            {
                DEVICE_B2B_H(ctx, aux);
               
#pragma unroll
                for ( ; ctx->c < BUF_SIZE_8 && j < NONCE_SIZE_8; ++j)
                {
                    ctx->b[ctx->c++] = ((uint8_t *)non)[NONCE_SIZE_8 - j - 1];
                }
            }

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
			uint8_t iii[8];
			iii[0] = ((uint8_t *)(&h3))[3];
			iii[1] = ((uint8_t *)(&h3))[2];
			iii[2] = ((uint8_t *)(&h3))[1];
            iii[3] = ((uint8_t *)(&h3))[0];

           // printf( " \n -----------2------------- ");
           // for (int kj = 0; kj<32; kj++)
          //  printf("\n %d %d ",kj , ((uint8_t*)r)[kj]);

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

        //====================================================================//
        //  Hash 
        //====================================================================//
#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < HEIGHT_SIZE; ++j)
        {
            ctx->b[ctx->c++] = iii[j];
        }

        //====================================================================//
        //  Hash height
        //====================================================================//
#pragma unroll
        for (j = 0; ctx->c < BUF_SIZE_8 && j < HEIGHT_SIZE; ++j)
        {
            ctx->b[ctx->c++] = ((const uint8_t *)&height)[j/*HEIGHT_SIZE - j - 1*/];
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
            HOST_B2B_H(ctx, aux);

            for (; ctx->c < BUF_SIZE_8 && j < CONST_MES_SIZE_8; ++j)
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



        HOST_B2B_H_LAST(ctx, aux);


#pragma unroll
        for (j = 0; j < NUM_SIZE_8; ++j)
        {
            ((uint8_t*)r)[j] = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
            //((uint8_t*)r)[NUM_SIZE_8 - j - 1] = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
        }

	//			printf("\n ---------3----------");
	//			for (int kj = 0; kj<32; kj++)
	//				printf("\n %d %d ",kj , ((uint8_t*)r)[kj]);


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
				ctx->b[ctx->c++] = (( const uint8_t *)data)[j];
			}



			while (j < NUM_SIZE_8)
			{
				HOST_B2B_H(ctx, aux);

				while (ctx->c < BUF_SIZE_8 && j < NUM_SIZE_8)
				{
					ctx->b[ctx->c++] = (( const uint8_t *)data)[j++];
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
				//((uint8_t*)r)[NUM_SIZE_8 - j - 1] = (ctx->h[j >> 3] >> ((j & 7) << 3)) & 0xFF;
            }

//printf("\n --------4-----------");
//				for (int kj = 0; kj<32; kj++)
//					printf("\n %d %d ",kj , ((uint8_t*)r)[kj]);
            
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

//printf("\n ----indx-----");
//			for(int p=0;p<32;p++)
//					printf("\n %d" ,ind[p]);  

            //================================================================//
            //  Calculate result
            //================================================================//

//			for(int g = 0; g < 32 ; g ++)
///			{
//			 for (j = 0; j < 32; ++j)
//			 {
//
//			    printf("\n index: %d , j: %d ,  %d ",g , j , ((uint8_t *)&hashes[ind[g] << 3])[j]);
///			 }
///			}
//            // first addition of hashes -> r
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


            

					for (int i = 0; i < NUM_SIZE_8; ++i)
					{
						((uint8_t *)&pre_r)[i] = ((uint8_t *)&r)[NUM_SIZE_8-i-1];
					}
/*
			if (tid == 0)
			{
				printf("\n  ----r---------- ");

				for (int kj = 0; kj<32; kj++)
					printf("\n %d %d ",kj , ((uint8_t*)r)[kj]);

			}
*/
			//--------------------hash f
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


			//--------------hash 
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

            
  //          printf("\n j: %d",j);
            if(j )
            {
    
                
                 //   uint32_t id = atomicAdd(count, 1);
	         //   if(id == 0)
{
                    valid[0] = tid+1; 
                    #pragma unroll
                    for (int i = 0; i < NUM_SIZE_32; ++i)
                    {
                        res[i] = r[i];

                    }
/*
printf("\n --r-- ");
                    for(int i=0 ; i< 32;i++)
{
printf("%d , %d",i,r[i]);
}
printf("\n --d--");
for(int i=0;i<32;i++)
{
printf("\n %d , %d ", i , bound[i]);
}
*/

 }
           }
        }



    return;

}

// mining.cu


