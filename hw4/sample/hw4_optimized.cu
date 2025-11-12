//***********************************************************************************
// CUDA-Optimized Bitcoin Block Miner
// Parallelizes nonce search across GPU threads
// Performance improvements:
//   - Parallel nonce computation (100-500x speedup expected)
//   - Device-side SHA256 computation
//   - Constant memory for block parameters
//   - Early termination with atomics
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>

#include "sha256.h"

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 4096
#define TOTAL_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;

////////////////////////   Device Constants   /////////////////////

__constant__ unsigned int d_k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__constant__ unsigned char d_target_hex[32];
__constant__ HashBlock d_block_template;

////////////////////////   Utils (Device)   ///////////////////////

__device__ inline unsigned int rotr32(unsigned int x, int n)
{
    return (x >> n) | (x << (32 - n));
}

__device__ int little_endian_bit_comparison_device(const unsigned int *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit (little-endian)
    for(int i = byte_len - 1; i >= 0; --i)
    {
        unsigned char a_byte = ((unsigned char*)a)[i];
        unsigned char b_byte = b[i];
        if(a_byte < b_byte)
            return -1;
        else if(a_byte > b_byte)
            return 1;
    }
    return 0;
}

////////////////////////   Device SHA256   ///////////////////////

typedef struct
{
    unsigned int h[8];
} SHA256_Device;

__device__ void sha256_transform_device(SHA256_Device *ctx, const unsigned char *msg)
{
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int w[64];

    // Copy chunk into first 16 words
    for(int i = 0; i < 16; ++i)
    {
        int j = i * 4;
        w[i] = ((unsigned int)msg[j] << 24) | 
               ((unsigned int)msg[j+1] << 16) | 
               ((unsigned int)msg[j+2] << 8) | 
               ((unsigned int)msg[j+3]);
    }

    // Extend the first 16 words into the remaining 48 words
    for(int i = 16; i < 64; ++i)
    {
        unsigned int s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        unsigned int s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    // Initialize working variables
    a = ctx->h[0];
    b = ctx->h[1];
    c = ctx->h[2];
    d = ctx->h[3];
    e = ctx->h[4];
    f = ctx->h[5];
    g = ctx->h[6];
    h = ctx->h[7];

    // Compression main loop
    for(int i = 0; i < 64; ++i)
    {
        unsigned int S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        unsigned int S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int temp1 = h + S1 + ch + d_k[i] + w[i];
        unsigned int temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;
}

__device__ void sha256_device(SHA256_Device *ctx, const unsigned char *msg, size_t len)
{
    // Initialize hash values
    ctx->h[0] = 0x6a09e667;
    ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372;
    ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f;
    ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab;
    ctx->h[7] = 0x5be0cd19;

    size_t remain = len % 64;
    size_t total_len = len - remain;

    // Process message in 512-bit chunks
    for(size_t i = 0; i < total_len; i += 64)
    {
        sha256_transform_device(ctx, &msg[i]);
    }

    // Process remaining data
    unsigned char m[64];
    for(int i = 0; i < 64; ++i) m[i] = 0;

    size_t j = 0;
    for(size_t i = total_len; i < len; ++i, ++j)
    {
        m[j] = msg[i];
    }

    m[j++] = 0x80;

    if(j > 56)
    {
        sha256_transform_device(ctx, m);
        for(int i = 0; i < 64; ++i) m[i] = 0;
        j = 0;
    }

    unsigned long long L = len * 8;
    m[63] = L;
    m[62] = L >> 8;
    m[61] = L >> 16;
    m[60] = L >> 24;
    m[59] = L >> 32;
    m[58] = L >> 40;
    m[57] = L >> 48;
    m[56] = L >> 56;

    sha256_transform_device(ctx, m);

    // NO byte swapping here - keep as is from transform
}

__device__ void double_sha256_device(unsigned char *result, const unsigned char *data, size_t len)
{
    SHA256_Device tmp, final;
    
    // First SHA256
    sha256_device(&tmp, data, len);
    unsigned char tmp_bytes[32];
    for(int i = 0; i < 8; ++i)
    {
        unsigned int h = tmp.h[i];
        tmp_bytes[i*4 + 0] = (h >> 24) & 0xff;
        tmp_bytes[i*4 + 1] = (h >> 16) & 0xff;
        tmp_bytes[i*4 + 2] = (h >> 8) & 0xff;
        tmp_bytes[i*4 + 3] = h & 0xff;
    }
    
    // Second SHA256
    sha256_device(&final, tmp_bytes, 32);
    
    // Convert final hash to bytes
    for(int i = 0; i < 8; ++i)
    {
        unsigned int h = final.h[i];
        result[i*4 + 0] = (h >> 24) & 0xff;
        result[i*4 + 1] = (h >> 16) & 0xff;
        result[i*4 + 2] = (h >> 8) & 0xff;
        result[i*4 + 3] = h & 0xff;
    }
}

////////////////////////   Mining Kernel   ///////////////////////

__global__ void mining_kernel(unsigned int *found_nonce, unsigned char *found_hash)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nonces_per_thread = (0xffffffffU + 1) / TOTAL_THREADS;
    unsigned int nonce_start = tid * nonces_per_thread;
    unsigned int nonce_end = nonce_start + nonces_per_thread;

    // Last thread handles remaining nonces
    if(tid == TOTAL_THREADS - 1)
    {
        nonce_end = 0xffffffffU + 1;
    }

    HashBlock block = d_block_template;
    unsigned char hash_result[32];

    for(unsigned int nonce = nonce_start; nonce < nonce_end; ++nonce)
    {
        block.nonce = nonce;
        double_sha256_device(hash_result, (unsigned char*)&block, sizeof(block));

        // Check if hash < target (little-endian comparison)
        if(little_endian_bit_comparison_device((unsigned int*)hash_result, d_target_hex, 32) < 0)
        {
            // Found solution - use atomic to ensure thread safety
            atomicCAS(found_nonce, 0xffffffffU, nonce);
            
            // Copy hash result
            for(int i = 0; i < 32; ++i)
            {
                found_hash[i] = hash_result[i];
            }
            return;
        }

        // Early exit if another thread found solution
        if(*found_nonce != 0xffffffffU)
        {
            return;
        }
    }
}

////////////////////////   Utils (Host)   ///////////////////////

unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a': return 0x0a;
        case 'b': return 0x0b;
        case 'c': return 0x0c;
        case 'd': return 0x0d;
        case 'e': return 0x0e;
        case 'f': return 0x0f;
        case '0' ... '9': return c - '0';
    }
    return 0;
}

void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);
    size_t b = string_len / 2 - 1;
    for(size_t s = 0; s < string_len; s += 2, --b)
    {
        out[b] = (unsigned char)(decode(in[s]) << 4) + decode(in[s+1]);
    }
}

void print_hex(unsigned char* hex, size_t len)
{
    for(size_t i = 0; i < len; ++i)
    {
        printf("%02x", hex[i]);
    }
}

void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i = len - 1; i >= 0; --i)
    {
        printf("%02x", hex[i]);
    }
}

void getline(char *str, size_t len, FILE *fp)
{
    int i = 0;
    while(i < len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Host Solve Function   ///////////////////////

void solve(FILE *fin, FILE *fout)
{
    // Read block data
    char version[9], prevhash[65], ntime[9], nbits[9];
    int tx;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);

    char *raw_merkle_branch = new char[tx * 65];
    char **merkle_branch = new char*[tx];
    for(int i = 0; i < tx; ++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // Calculate merkle root using host CPU function
    unsigned char merkle_root[32];
    
    // Simple merkle root calculation (keeping host version for now)
    size_t total_count = tx;
    unsigned char *raw_list = new unsigned char[(total_count + 1) * 32];
    unsigned char **list = new unsigned char*[total_count + 1];

    for(int i = 0; i < total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        convert_string_to_little_endian_bytes(list[i], merkle_branch[i], 64);
    }
    list[total_count] = raw_list + total_count * 32;

    while(total_count > 1)
    {
        int i, j;
        if(total_count % 2 == 1)
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i = 0, j = 0; i < total_count; i += 2, ++j)
        {
            SHA256 tmp;
            sha256(&tmp, (BYTE*)list[i], 64);
            SHA256 sha256_ctx;
            sha256(&sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
            memcpy(list[j], sha256_ctx.b, 32);
        }

        total_count = j;
    }

    memcpy(merkle_root, list[0], 32);
    delete[] raw_list;
    delete[] list;

    // Prepare block structure
    HashBlock block_host;
    convert_string_to_little_endian_bytes((unsigned char*)&block_host.version, version, 8);
    convert_string_to_little_endian_bytes(block_host.prevhash, prevhash, 64);
    memcpy(block_host.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char*)&block_host.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char*)&block_host.ntime, ntime, 8);
    block_host.nonce = 0;

    // Calculate target value
    unsigned int exp = block_host.nbits >> 24;
    unsigned int mant = block_host.nbits & 0xffffff;
    unsigned char target_hex[32] = {};

    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;

    target_hex[sb]     = (mant << rb);
    target_hex[sb + 1] = (mant >> (8 - rb));
    target_hex[sb + 2] = (mant >> (16 - rb));
    target_hex[sb + 3] = (mant >> (24 - rb));

    printf("Block info (big):\n");
    printf("  version:   %s\n", version);
    printf("  prevhash:  %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  nbits:     %s\n", nbits);
    printf("  ntime:     %s\n", ntime);
    printf("  Target:    "); print_hex_inverse(target_hex, 32); printf("\n");
    printf("  nonce:     ???\n\n");

    // Copy data to device constant memory
    cudaMemcpyToSymbol(d_block_template, &block_host, sizeof(HashBlock));
    cudaMemcpyToSymbol(d_target_hex, target_hex, 32);

    // Allocate device memory for results
    unsigned int *d_found_nonce;
    unsigned char *d_found_hash;
    cudaMalloc(&d_found_nonce, sizeof(unsigned int));
    cudaMalloc(&d_found_hash, 32);

    // Initialize result values
    unsigned int h_found_nonce = 0xffffffffU;
    unsigned char h_found_hash[32] = {};
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch kernel
    printf("Launching mining kernel: %d blocks, %d threads/block\n", BLOCKS_PER_GRID, THREADS_PER_BLOCK);
    mining_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_found_nonce, d_found_hash);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_found_hash, d_found_hash, 32, cudaMemcpyDeviceToHost);

    printf("hash(little): ");
    print_hex(h_found_hash, 32);
    printf("\n");

    printf("hash(big):    ");
    print_hex_inverse(h_found_hash, 32);
    printf("\n\n");

    // Output nonce
    for(int i = 0; i < 4; ++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&h_found_nonce)[i]);
    }
    fprintf(fout, "\n");

    // Cleanup
    cudaFree(d_found_nonce);
    cudaFree(d_found_hash);
    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

////////////////////////   Main   ///////////////////////

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    if(!fin || !fout)
    {
        fprintf(stderr, "Error opening files\n");
        return 1;
    }

    int totalblock;
    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for(int i = 0; i < totalblock; ++i)
    {
        printf("\n========== Block %d ==========\n", i);
        solve(fin, fout);
    }

    fclose(fin);
    fclose(fout);

    return 0;
}
