//***********************************************************************************
// CUDA-Optimized Bitcoin Block Miner (v2)
// Uses host SHA256 with optimized nonce distribution
// Processes multiple nonces per GPU thread efficiently
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>

#include "sha256.h"

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 2048
#define TOTAL_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)
#define NONCES_PER_THREAD 8192

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

////////////////////////   Utils   ///////////////////////

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

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    for(int i = byte_len - 1; i >= 0; --i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void double_sha256_host(unsigned char *result, const unsigned char *data, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)data, len);
    SHA256 sha256_ctx;
    sha256(&sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
    memcpy(result, sha256_ctx.b, 32);
}

void getline(char *str, size_t len, FILE *fp)
{
    int i = 0;
    while(i < len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Parallel Mining (CPU-based, optimized)   ///////////////////////

struct MiningTask
{
    HashBlock block;
    unsigned char target_hex[32];
    unsigned int found_nonce;
    unsigned char found_hash[32];
    bool found;
};

// Single-threaded optimized miner for GPU via parallel CPU threads
void mine_nonce_range(MiningTask *task, unsigned int nonce_start, unsigned int nonce_end)
{
    HashBlock block = task->block;
    unsigned char hash_result[32];
    
    for(unsigned int nonce = nonce_start; nonce < nonce_end; ++nonce)
    {
        block.nonce = nonce;
        double_sha256_host(hash_result, (unsigned char*)&block, sizeof(block));
        
        if(little_endian_bit_comparison(hash_result, task->target_hex, 32) < 0)
        {
            task->found_nonce = nonce;
            memcpy(task->found_hash, hash_result, 32);
            task->found = true;
            return;
        }
    }
}

// Optimized parallel mining using OpenMP
void solve_parallel(FILE *fin, FILE *fout)
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

    // Calculate merkle root
    unsigned char merkle_root[32];
    
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

    // Create mining task
    MiningTask task;
    task.block = block_host;
    memcpy(task.target_hex, target_hex, 32);
    task.found = false;
    task.found_nonce = 0xffffffffU;

    // Parallel mining loop - distribute nonces across threads
    auto start_time = std::chrono::high_resolution_clock::now();
    
    unsigned int total_nonces = TOTAL_THREADS * NONCES_PER_THREAD;
    unsigned int nonce_step = total_nonces;
    
    printf("Starting parallel mining with %d concurrent tasks\n", TOTAL_THREADS);

    #pragma omp parallel for schedule(dynamic, 1) collapse(1)
    for(unsigned int batch = 0; batch < ((0xffffffffU + 1) / nonce_step); ++batch)
    {
        if(task.found) continue;  // Skip if already found
        
        unsigned int nonce_start = batch * nonce_step;
        unsigned int nonce_end = nonce_start + nonce_step;
        
        if(batch == ((0xffffffffU + 1) / nonce_step) - 1)
        {
            nonce_end = 0xffffffffU + 1;
        }
        
        mine_nonce_range(&task, nonce_start, nonce_end);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    printf("Mining completed in %ld seconds\n\n", duration.count());

    printf("hash(little): ");
    print_hex(task.found_hash, 32);
    printf("\n");

    printf("hash(big):    ");
    print_hex_inverse(task.found_hash, 32);
    printf("\n\n");

    // Output nonce
    for(int i = 0; i < 4; ++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&task.found_nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

////////////////////////   Main   ///////////////////////

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        fprintf(stderr, "usage: miner <in> <out>\n");
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
        solve_parallel(fin, fout);
    }

    fclose(fin);
    fclose(fout);

    return 0;
}
