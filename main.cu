#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/*
extern "C" {
#include "crypto-algorithms/sha256.h"
}
*/

#include "SHA256CUDA/SHA256CUDA/sha256.cuh"

#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <string.h>

using namespace std;

cudaError_t crack(BYTE hash[SHA256_BLOCK_SIZE], char **wordlist, int word_count);

__device__ int dev_strlen(char* str) {
	int pos = 0;

	while(str[pos] != 0) {
		pos++;
	}

	return pos;
}

__device__ void dev_strncpy(char* dest, const char*  src, size_t size) {
	for(int i = 0; i < size; i++) {
		dest[i] = src[i];
	}
}

__device__ bool dev_cmphash(BYTE* first, BYTE* second, size_t size) {
	for(int i = 0; i < size; i++) {
		if(first[i] != second[i]) {
			return false;
		}
	}

	return true;
}


__device__ void dev_printBytes(BYTE* buf, size_t size) {
	for(int i = 0; i < size; i++) {
		printf("%02x", buf[i]);
	}

	printf("\n");
}

__global__ void crackKernel(BYTE hash[SHA256_BLOCK_SIZE], char
		**wordlist, int word_count)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x,
		thread_count = blockDim.x * gridDim.x,
		quotient = word_count / thread_count,
		thread_word_count = quotient + 1,
		start = thread_word_count * i,
		end = min(start + thread_word_count + 1, word_count - 1),
		/*
		remainder = word_count % thread_count,
		thread_word_count = quotient + (i > remainder ? 1 : 0),
		start = quotient * i + min(i, remainder),
		end = min(start + thread_word_count, word_count - 1),
		*/
		size = 0;

	for(int pos = start; pos <= end; pos++) {
		size = dev_strlen(wordlist[pos]);

		BYTE buf[SHA256_BLOCK_SIZE];

		SHA256_CTX ctx;

		sha256_init(&ctx);
		sha256_update(&ctx, (BYTE*)wordlist[pos], size);
		sha256_final(&ctx, buf);

		if(dev_cmphash(buf, hash, SHA256_BLOCK_SIZE)) {
			printf("Found word: %s\n", wordlist[pos]); 
		}
	}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int hexDigitToInt(char digit) {
	if(digit >= '0' && digit <= '9') {
		return digit - '0';
	} else if (digit >= 'a' && digit <= 'f') {
		return digit - 'a' + 10;
	} else if (digit >= 'A' && digit <= 'F') {
		return digit - 'A' + 10;
	} else {
		return 0;
	}
}

void hexStringToBytes(string hex, BYTE *buffer)
{
	int pos = 0, val = 0;

	while(pos < hex.size() - 1) {
		val = hexDigitToInt(hex[pos]) * 16;
		val += hexDigitToInt(hex[pos + 1]);

		buffer[pos / 2] = val;

		pos += 2;
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k,
				sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

int main(int argc, char **argv)
{
	cudaError_t cudaStatus;

	if(argc < 3) {
		printf("Usage: %s <worlist> <hashfile>\n", argv[0]);
		return 1;
	}

	ifstream hash_file(argv[2]);
	string hash_hex;

	hash_file >> hash_hex;

	BYTE *buf = (BYTE*)malloc(SHA256_BLOCK_SIZE * sizeof(BYTE));
	hexStringToBytes(hash_hex, buf);

	/*
	for(int i = 0; i < SHA256_BLOCK_SIZE; i++) {
		printf("%02x", buf[i]);
	}
	printf("\n");
	return 0;
	*/

	/*
	BYTE *text1;
	text1 = new BYTE[20];
	strncpy((char*)text1, "asd", 4);

	BYTE buf[SHA256_BLOCK_SIZE];
	SHA256_CTX ctx;

	sha256_init(&ctx);
	sha256_update(&ctx, text1, strlen((char*)text1));
	sha256_final(&ctx, buf);

	for(int i = 0; i < 32; ++i)
	{
		printf("%02x", buf[i]);
	}
	printf("\n");
	*/

	//char wordlist[3][100] = {"asd", "qwerty", "123456"};

	printf("Reading wordlist\n");

	ifstream file(argv[1]);
	
	vector<string> string_vector;
	string line;

	getline(file, line);
	while(!file.eof()) {
		string_vector.emplace_back(line);
		getline(file, line);
	}

	char** wordlist = (char**)malloc(string_vector.size() * sizeof(char*));

	for(int i = 0; i < string_vector.size(); i++) {
		wordlist[i] = (char*)malloc(string_vector[i].size() *
				sizeof(char));
		strncpy(wordlist[i], string_vector[i].c_str(),
				string_vector[i].size());
	}

	file.close();

	int word_count = string_vector.size();

	printf("Starting\n");

	crack(buf, (char**)wordlist, word_count);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t crack(BYTE hash[SHA256_BLOCK_SIZE], char **wordlist, int word_count)
{
	char **dev_wordlist = 0;
	char **dev_word_pointers = 0;
	BYTE *dev_hash = 0;
    cudaError_t cudaStatus;
	size_t word_length = 0;

	dev_word_pointers = new char*[word_count];

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	pre_sha256();

    cudaStatus = cudaMalloc((void**)&dev_wordlist, word_count *
			sizeof(char*));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	for(int i = 0; i < word_count; i++) {
		word_length = strlen(wordlist[i]) + 1;

		cudaStatus = cudaMalloc((void**)&dev_word_pointers[i],
				word_length * sizeof(char));

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "pointer cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_word_pointers[i], wordlist[i],
				word_length * sizeof(char), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "word memcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_wordlist + i,
				&dev_word_pointers[i], sizeof(char*),
				cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "pointer memcpy failed!");
			goto Error;
		}
	}

    cudaStatus = cudaMalloc((void**)&dev_hash, SHA256_BLOCK_SIZE *
			sizeof(BYTE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "hash cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_hash, hash, SHA256_BLOCK_SIZE * sizeof(BYTE),
			cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "hash memcpy failed! %s\n",
				cudaGetErrorString(cudaGetLastError()));
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	crackKernel<<<10, 196>>>(dev_hash, dev_wordlist, word_count);
	//crackKernel<<<1, 1>>>(dev_hash, dev_wordlist, word_count);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "crackKernel launch failed: %s, %d\n",
				cudaGetErrorString(cudaStatus), cudaStatus);
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	/*
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	*/

Error:

	for(int i = 0; i < word_count; i++) {
		cudaFree(dev_word_pointers[i]);
	}

	delete dev_word_pointers;

	cudaFree(dev_wordlist);
	cudaFree(dev_hash);
    
    return cudaStatus;
}
