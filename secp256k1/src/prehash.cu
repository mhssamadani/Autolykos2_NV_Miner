// prehash.cu

/*******************************************************************************

	PREHASH -- precalculation of hashes

*******************************************************************************/

#include "../include/prehash.h"
#include "../include/compaction.h"
#include "../include/definitions.h"
#include <cuda.h>


__device__ __forceinline__
static void store64(uint64_t *res64,const  uint64_t num64)
{
	((uint32_t* )res64)[0] = __byte_perm(((uint32_t*)(&num64))[1], 0, 0x0123);
	((uint32_t* )res64)[1] = __byte_perm(((uint32_t*)(&num64))[0], 0, 0x0123);
}
__device__ __forceinline__ uint32_t ld_gbl_cs(const  uint32_t * __restrict__ p) {
	uint32_t v;
	asm("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
	return v;
}

__device__ __forceinline__ uint4 ld_gbl_cs_v4(const  uint4 * __restrict__ p) {
	uint4 v;
	asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
	return v;
}

__device__ __forceinline__ uint32_t cuda_swab32(uint32_t x)
{
	/* device */
	return __byte_perm(x, x, 0x0123);
}

__device__ __forceinline__ uint64_t devectorize(uint2 x)
{
	uint64_t result;
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(result) : "r"(x.x), "r"(x.y));
	return result;
}


__device__ __forceinline__ uint2 vectorize(const uint64_t x)
{
	uint2 result;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.x), "=r"(result.y) : "l"(x));
	return result;
}

__device__ __forceinline__
uint64_t devROTR64(uint64_t b, int offset)
{
	uint2 a;
	uint2 result;
	a = vectorize(b);

	if (offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return devectorize(result);
}



__constant__  uint8_t blake2b_sigma[12][16] = {
 {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
 { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
 { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
 {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
 {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
 {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
 { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
 { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
 {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
 { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
 {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
 { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};


#define G(m, r, i, a, b, c, d) do { \
	a += b + ((uint64_t *)m)[blake2b_sigma[r][i]]; \
	d = devROTR64(d ^ a, 32); \
	c += d; \
	b = devROTR64(b ^ c, 24); \
	a += b + ((uint64_t *)m)[blake2b_sigma[r][i + 1]]; \
	d = devROTR64(d ^ a, 16); \
	c += d; \
	b =  devROTR64(b ^ c, 63); \
} while(0)


#define BLAKE2B_RND(v, r, m) do { \
	G(m, r, 0, v[ 0], v[ 4], v[ 8], v[12]); \
	G(m, r, 2, v[ 1], v[ 5], v[ 9], v[13]); \
	G(m, r, 4, v[ 2], v[ 6], v[10], v[14]); \
	G(m, r, 6, v[ 3], v[ 7], v[11], v[15]); \
	G(m, r, 8, v[ 0], v[ 5], v[10], v[15]); \
	G(m, r, 10, v[ 1], v[ 6], v[11], v[12]); \
	G(m, r, 12, v[ 2], v[ 7], v[ 8], v[13]); \
	G(m, r, 14, v[ 3], v[ 4], v[ 9], v[14]); \
} while(0)


__device__ __forceinline__ void BlakeCompress(uint64_t  *h, const uint64_t  *m, uint64_t  t, uint64_t  f)
{
	uint64_t  v[16];

	v[0] = h[0];
	v[1] = h[1];
	v[2] = h[2];
	v[3] = h[3];
	v[4] = h[4];
	v[5] = h[5];
	v[6] = h[6];
	v[7] = h[7];

	v[8] = 0x6A09E667F3BCC908UL;
	v[9] = 0xBB67AE8584CAA73BUL;
	v[10] = 0x3C6EF372FE94F82BUL;
	v[11] = 0xA54FF53A5F1D36F1UL;
	v[12] = 0x510E527FADE682D1UL;
	v[13] = 0x9B05688C2B3E6C1FUL;
	v[14] = 0x1F83D9ABFB41BD6BUL;
	v[15] = 0x5BE0CD19137E2179UL;

	v[12] ^= t;
	v[14] ^= f;

#pragma unroll
	for (int rnd = 0; rnd < 12; ++rnd)
	{
		BLAKE2B_RND(v, rnd, m);
	}

	h[0] ^= v[0] ^ v[0 + 8];
	h[1] ^= v[1] ^ v[1 + 8];
	h[2] ^= v[2] ^ v[2 + 8];
	h[3] ^= v[3] ^ v[3 + 8];
	h[4] ^= v[4] ^ v[4 + 8];
	h[5] ^= v[5] ^ v[5 + 8];
	h[6] ^= v[6] ^ v[6 + 8];
	h[7] ^= v[7] ^ v[7 + 8];
}


////////////////////////////////////////////////////////////////////////////////
//  Precalculate hashes
////////////////////////////////////////////////////////////////////////////////
int Prehash(
	uint64_t N_LEN ,
	uint32_t * hashes,
	uint32_t  height
)
{

	
	InitPrehash << <1 + (N_LEN - 1) / BLOCK_DIM, BLOCK_DIM >> > (
		N_LEN,height, hashes
		);
	CUDA_CALL(cudaPeekAtLastError());

	return EXIT_SUCCESS;

}
__global__ void InitPrehash(
	const uint32_t  n_len,
	// height
	const uint32_t  height,
	// hashes
	uint32_t * hashes
)
{
	uint32_t tid = threadIdx.x;

	tid += blockDim.x * blockIdx.x;


	if (tid < n_len)
	{

		//====================================================================//
		//  Initialize context
		//====================================================================//

		uint64_t h[8] = { 0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL, 0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL, 0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL, 0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL };
		uint64_t b[16];
		uint64_t t = 0;

		h[0] ^= 0x01010020;

		//====================================================================//
		//  Hash tid
		//====================================================================//
		((uint32_t *)b)[0] = __byte_perm(tid, tid, 0x0123);

		//====================================================================//
		//  Hash height
		//====================================================================//
		((uint32_t *)b)[1] = height;

		//====================================================================//
		//  Hash constant message
		//====================================================================//
		uint64_t ctr = 0;
		for (int x = 1; x < 16; ++x, ++ctr)
		{
			store64(&(b[x]), ctr);
		}
#pragma unroll 1
		for (int z = 0; z < 63; ++z)
		{
			t += 128;
			BlakeCompress((uint64_t *)h, (uint64_t *)b, t, 0UL);

#pragma unroll
			for (int x = 0; x < 16; ++x, ++ctr)
			{
				store64(&(b[x]), ctr);
			}
		}
		t += 128;
		BlakeCompress((uint64_t *)h, (uint64_t *)b, t, 0UL);

		store64(&(b[0]), ctr);

		t += 8;

#pragma unroll
		for (int i = 1; i < 16; ++i) ((uint64_t *)b)[i] = 0UL;

		BlakeCompress((uint64_t *)h, (uint64_t *)b, t, 0xFFFFFFFFFFFFFFFFUL);


		//====================================================================//
		//  Dump result to global memory -- BIG ENDIAN
		//====================================================================//

#pragma unroll
		for (int i = 0; i < 4; ++i) store64(&(((uint64_t *)hashes)[(tid + 1) * 4 - i - 1]), h[i]);
		((uint8_t *)hashes)[tid * 32 + 31] = 0;
	}

	return;
}
// prehash.cu





