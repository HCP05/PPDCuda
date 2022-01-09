#include <string.h>
#include <stdio.h>
#include "crypto-algorithms/sha256.h"
//#include <openssl/evp.h>

/*
char computeHash(char *unhashed, char* hashed)
{
    char success = 0;

    EVP_MD_CTX* context = EVP_MD_CTX_new();

    if(context != NULL)
    {
        if(EVP_DigestInit_ex(context, EVP_sha256(), NULL))
        {
            if(EVP_DigestUpdate(context, unhashed, strlen(unhashed)))
            {
                unsigned char hash[EVP_MAX_MD_SIZE];
                unsigned int lengthOfHash = 0;

                if(EVP_DigestFinal_ex(context, hash, &lengthOfHash))
                {
					memcpy(hashed, hash, lengthOfHash);

                    success = 1;
                }
            }
        }

        EVP_MD_CTX_free(context);
    }

    return success;
}
*/

int main(int, char**)
{
	/*
	unsigned char digest[32];

	computeHash("asd", digest);
	*/

	BYTE text1[] = {"asd"};
	BYTE buf[SHA256_BLOCK_SIZE];
	SHA256_CTX ctx;

	sha256_init(&ctx);
	sha256_update(&ctx, text1, strlen(text1));
	sha256_final(&ctx, buf);

	for(int i = 0; i < 32; ++i)
	{
		printf("%02x", buf[i]);
	}
	printf("\n");

    return 0;
}
