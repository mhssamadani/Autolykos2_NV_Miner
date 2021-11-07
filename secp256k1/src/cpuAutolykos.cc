#include "../include/cpuAutolykos.h"
#include "../include/request.h"

AutolykosAlg::AutolykosAlg()
{
	m_str = new char[64];
	bound_str = new char[100];
	m_n = new uint8_t[NUM_SIZE_8 + NONCE_SIZE_8];
	p_w_m_n = new uint8_t[PK_SIZE_8 + PK_SIZE_8 + NUM_SIZE_8 + NONCE_SIZE_8];
	Hinput = new uint8_t[sizeof(uint32_t) + CONST_MES_SIZE_8 + PK_SIZE_8 + NUM_SIZE_8 + PK_SIZE_8];
	n_str = new char[NONCE_SIZE_4];
	h_str = new char[HEIGHT_SIZE];


	int tr = sizeof(unsigned long long);
	for (size_t i = 0; i < CONST_MES_SIZE_8 / tr; i++)
	{
		unsigned long long tmp = i;
		uint8_t tmp2[8];
		uint8_t tmp1[8];
		memcpy(tmp1, &tmp, tr);
		tmp2[0] = tmp1[7];
		tmp2[1] = tmp1[6];
		tmp2[2] = tmp1[5];
		tmp2[3] = tmp1[4];
		tmp2[4] = tmp1[3];
		tmp2[5] = tmp1[2];
		tmp2[6] = tmp1[1];
		tmp2[7] = tmp1[0];
		memcpy(&CONST_MESS[i], tmp2, tr);
	}
}


AutolykosAlg::~AutolykosAlg()
{

}


void AutolykosAlg::Blake2b256(const char * in,
	const int len,
	uint8_t * output,
	char * outstr)
{
	ctx_t ctx;
	uint64_t aux[32];

	//====================================================================//
	//  Initialize context
	//====================================================================//
	memset(ctx.b, 0, 128);
	B2B_IV(ctx.h);
	ctx.h[0] ^= 0x01010000 ^ NUM_SIZE_8;
	memset(ctx.t, 0, 16);
	ctx.c = 0;

	//====================================================================//
	//  Hash message
	//====================================================================//
	for (int i = 0; i < len; ++i)
	{
		if (ctx.c == 128) { HOST_B2B_H(&ctx, aux); }

		ctx.b[ctx.c++] = (uint8_t)(in[i]);
	}
	HOST_B2B_H_LAST(&ctx, aux);
	for (int i = 0; i < NUM_SIZE_8; ++i)
	{
		output[NUM_SIZE_8 - i - 1] = (ctx.h[i >> 3] >> ((i & 7) << 3)) & 0xFF;
	}

	LittleEndianToHexStr(output, NUM_SIZE_8, outstr);

}



void AutolykosAlg::GenIdex(const char * in, const int len, uint32_t* index, uint64_t N_LEN)
{
	int a = INDEX_SIZE_8;
	int b = K_LEN;
	int c = NUM_SIZE_8;
	int d = NUM_SIZE_4;


	uint8_t sk[NUM_SIZE_8 * 2];
	char skstr[NUM_SIZE_4 + 10];

	memset(sk, 0, NUM_SIZE_8 * 2);
	memset(skstr, 0, NUM_SIZE_4);

	Blake2b256(in, len, sk, skstr);

	uint8_t beH[PK_SIZE_8];
	HexStrToBigEndian(skstr, NUM_SIZE_4, beH, NUM_SIZE_8);


	uint32_t* ind = index;

	memcpy(sk, beH, NUM_SIZE_8);
	memcpy(sk + NUM_SIZE_8, beH, NUM_SIZE_8);

	uint32_t tmpInd[32];
	int sliceIndex = 0;
	for (int k = 0; k < K_LEN; k++)
	{


		uint8_t tmp[4];
		memcpy(tmp, sk + sliceIndex, 4);
		memcpy(&tmpInd[k], sk + sliceIndex, 4);
		uint8_t tmp2[4];
		tmp2[0] = tmp[3];
		tmp2[1] = tmp[2];
		tmp2[2] = tmp[1];
		tmp2[3] = tmp[0];
		memcpy(&ind[k], tmp2, 4);
		ind[k] = ind[k] % N_LEN;
		sliceIndex++;
	}

}


void AutolykosAlg::hashFn(const char * in, const int len, uint8_t * output)
{
	char *skstr = new char[len * 3];
	Blake2b256(in, len, output, skstr);
	uint8_t beHash[PK_SIZE_8];
	HexStrToBigEndian(skstr, NUM_SIZE_4, beHash, NUM_SIZE_8);
	memcpy(output, beHash, NUM_SIZE_8);
	delete skstr;
}

bool AutolykosAlg::RunAlg(
	uint8_t *message,
	uint8_t *nonce,
	uint8_t *bPool,
	uint8_t *height
	)
{

	BigEndianToHexStr(message, NUM_SIZE_8, m_str);

	uint32_t ilen = 0;
	LittleEndianOf256ToDecStr((uint8_t *)bPool, bound_str, &ilen);


	uint32_t index[K_LEN];
	LittleEndianToHexStr(nonce, NONCE_SIZE_8, n_str);

	BigEndianToHexStr(height, HEIGHT_SIZE, h_str);
	uint8_t beN[NONCE_SIZE_8];
	HexStrToBigEndian(n_str, NONCE_SIZE_8 * 2, beN, NONCE_SIZE_8);

	uint8_t beH[HEIGHT_SIZE];
	HexStrToBigEndian(h_str, HEIGHT_SIZE * 2, beH, HEIGHT_SIZE);


	uint8_t h1[NUM_SIZE_8];
	memcpy(m_n, message, NUM_SIZE_8);
	memcpy(m_n + NUM_SIZE_8, beN, NONCE_SIZE_8);
	hashFn((const char *)m_n, NUM_SIZE_8 + NONCE_SIZE_8, (uint8_t *)h1);

	uint64_t h2;
	char tmpL1[8];
	tmpL1[0] = h1[31];
	tmpL1[1] = h1[30];
	tmpL1[2] = h1[29];
	tmpL1[3] = h1[28];
	tmpL1[4] = h1[27];
	tmpL1[5] = h1[26];
	tmpL1[6] = h1[25];
	tmpL1[7] = h1[24];
	memcpy(&h2, tmpL1, 8);

	uint32_t HH;
	memcpy(&HH,beH,HEIGHT_SIZE);
	uint32_t N_LEN = calcN(HH);
	unsigned int h3 = h2 % N_LEN;
	

	uint8_t iii[4];
	iii[0] = ((char *)(&h3))[3];
	iii[1] = ((char *)(&h3))[2];
	iii[2] = ((char *)(&h3))[1];
	iii[3] = ((char *)(&h3))[0];

	uint8_t i_h_M[HEIGHT_SIZE + HEIGHT_SIZE + CONST_MES_SIZE_8];
	memcpy(i_h_M, iii, HEIGHT_SIZE);
	memcpy(i_h_M + HEIGHT_SIZE, beH, HEIGHT_SIZE);
	memcpy(i_h_M + HEIGHT_SIZE + HEIGHT_SIZE, CONST_MESS, CONST_MES_SIZE_8);
	hashFn((const char *)i_h_M, HEIGHT_SIZE + HEIGHT_SIZE + CONST_MES_SIZE_8, (uint8_t *)h1);
	uint8_t ff[NUM_SIZE_8 - 1];
	memcpy(ff, h1 + 1, NUM_SIZE_8 - 1);


	uint8_t seed[NUM_SIZE_8 - 1 + NUM_SIZE_8 + NONCE_SIZE_8];
	memcpy(seed, ff, NUM_SIZE_8 - 1);
	memcpy(seed + NUM_SIZE_8 - 1, message, NUM_SIZE_8);
	memcpy(seed + NUM_SIZE_8 - 1 + NUM_SIZE_8, beN, NONCE_SIZE_8);
	GenIdex((const char*)seed, NUM_SIZE_8 - 1 + NUM_SIZE_8 + NONCE_SIZE_8, index,N_LEN);




	uint8_t ret[32][NUM_SIZE_8];
	int ll = sizeof(uint32_t) + CONST_MES_SIZE_8 + PK_SIZE_8 + NUM_SIZE_8 + PK_SIZE_8;


	BIGNUM* bigsum = BN_new();
	CALL(BN_dec2bn(&bigsum, "0"), ERROR_OPENSSL);

	BIGNUM* bigres = BN_new();
	CALL(BN_dec2bn(&bigres, "0"), ERROR_OPENSSL);

	int rep = 0;
	int off = 0;
	uint8_t tmp[NUM_SIZE_8 - 1];
	char hesStr[64 + 1];
	uint8_t tmp2[4];
	uint8_t tmp1[4];

	unsigned char f[32];
	memset(f, 0, 32);

	char *LSUMM;
	char *LB;
	for (rep = 0; rep < 32; rep++)
	{
		memset(Hinput, 0, ll);


		memcpy(tmp1, &index[rep], 4);
		tmp2[0] = tmp1[3];
		tmp2[1] = tmp1[2];
		tmp2[2] = tmp1[1];
		tmp2[3] = tmp1[0];

		off = 0;
		memcpy(Hinput + off, tmp2, sizeof(uint32_t));
		off += sizeof(uint32_t);

		memcpy(Hinput + off, beH, HEIGHT_SIZE);
		off += HEIGHT_SIZE;

		memcpy(Hinput + off, CONST_MESS, CONST_MES_SIZE_8);
		off += CONST_MES_SIZE_8;

		hashFn((const char *)Hinput, off, (uint8_t *)ret[rep]);

		memcpy(tmp, &(ret[rep][1]), 31);


		CALL(BN_bin2bn((const unsigned char *)tmp, 31, bigres), ERROR_OPENSSL);

		CALL(BN_add(bigsum, bigsum, bigres), ERROR_OPENSSL);

		LB = BN_bn2dec(bigres);

		BN_bn2bin(bigsum, f);


	}

	const char *SUMMbigEndian = BN_bn2dec(bigsum);

	BN_bn2bin(bigsum, f);
	char bigendian2littl[32];
	for (size_t i = 0; i < 32; i++)
	{
		bigendian2littl[i] = f[32 - i - 1];
	}

	BIGNUM* littleF = BN_new();
	CALL(BN_bin2bn((const unsigned char *)bigendian2littl, 32, littleF), ERROR_OPENSSL);
	const char *SUMMLittleEndian = BN_bn2dec(littleF);

	char hf[32];
	hashFn((const char *)f, 32, (uint8_t *)hf);

	BIGNUM* bigHF = BN_new();
	CALL(BN_bin2bn((const unsigned char *)hf, 32, bigHF), ERROR_OPENSSL);

	char littl2big[32];
	for (size_t i = 0; i < 32; i++)
	{
		littl2big[i] = bPool[32 - i - 1];
	}

	BIGNUM* bigB = BN_new();
	CALL(BN_bin2bn((const unsigned char *)littl2big, 32, bigB), ERROR_OPENSSL);

	int cmp = BN_cmp(bigHF, bigB);

	const char *chD = BN_bn2dec(bigHF);
	const char *chB = BN_bn2dec(bigB);


	BN_free(bigsum);
	BN_free(bigres);
	BN_free(littleF);
	BN_free(bigHF);
	BN_free(bigB);

	if (cmp < 0)
	{
		LOG(INFO) << "sol passed";	
		return true;
	}
	else
	{
		LOG(INFO) << "sol not passed";
		return false;
	}


}

