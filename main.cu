#include "cuda_runtime.h"
#include "device_launch_parameters.h"
extern "C" {
#include "crypto-algorithms/sha256.h"
}

#include <stdio.h>
//t
cudaError_t crack(BYTE hash[SHA256_BLOCK_SIZE], char **wordlist, int word_count);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void crackKernel(BYTE hash[SHA256_BLOCK_SIZE], char
		**wordlist, int word_count)
{
	int i = threadIdx.x;
	//printf("Thread id: %d\n", i);

	if(i < word_count) {
		printf("%s\n", wordlist[i]);
	}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
	cudaError_t cudaStatus;

	BYTE *text1;
	text1 = new BYTE[20];
	strncpy((char*)text1, "asd", 4);

	BYTE buf[SHA256_BLOCK_SIZE];
	SHA256_CTX ctx;

	sha256_init(&ctx);
	sha256_update(&ctx, text1, strlen((char*)text1));
	sha256_final(&ctx, buf);

	/*
	for(int i = 0; i < 32; ++i)
	{
		printf("%02x", buf[i]);
	}
	printf("\n");
	*/

	//char wordlist[3][100] = {"asd", "qwerty", "123456"};

	char** wordlist = (char**)malloc(3 * sizeof(char*));
	wordlist[0] = (char*)malloc(10 * sizeof(char*));
	wordlist[1] = (char*)malloc(10 * sizeof(char*));
	wordlist[2] = (char*)malloc(10 * sizeof(char*));
	strcpy(wordlist[0], "asd");
	strcpy(wordlist[1], "qwerty");
	strcpy(wordlist[2], "123456");

	int word_count = 3;

	crack(buf, (char**)wordlist, word_count);

	delete text1;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;

	/*
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
	*/
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

    cudaStatus = cudaMalloc((void**)&dev_hash, word_count *
			sizeof(BYTE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaMemcpy(dev_hash, hash, SHA256_BLOCK_SIZE * sizeof(BYTE),
			cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
	crackKernel<<<3, 196>>>(dev_hash, dev_wordlist, word_count);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "crackKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
