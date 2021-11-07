#pragma once

#include "definitions.h"
#include "easylogging++.h"



#include <curl/curl.h>
#include <inttypes.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>
#include <random>


#include "cryptography.h"
#include "conversion.h"
#include "definitions.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/opensslv.h>
#include <random>


class AutolykosAlg
{
public:
	unsigned long long CONST_MESS[CONST_MES_SIZE_8 / 8];

	AutolykosAlg();
	~AutolykosAlg();
	int m_iAlgVer;
	void Blake2b256(const char * in, const int len, uint8_t * output, char * outstr);
	void GenIdex(const char * in, const int len, uint32_t* index , uint64_t N_LEN);
	void hashFn(const char * in, const int len, uint8_t * output);
	bool RunAlg(
		uint8_t *B_mes,
		uint8_t *nonce,
		uint8_t *bPool,
		uint8_t *height
		);

private:
	char *m_str = NULL;
	char *bound_str = NULL;
	uint8_t *m_n;
	uint8_t  *p_w_m_n;
	uint8_t  *Hinput;
	char *n_str;
	char *h_str;
};


