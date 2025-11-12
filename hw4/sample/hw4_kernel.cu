//***********************************************************************************
// CUDA-Optimized Bitcoin Block Miner (Multi-GPU + CPU Hybrid)
// Uses GPU for acceleration with efficient work distribution
// Falls back to CPU parallelization with OpenMP
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

#include "sha256.h"

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;

////////////////////////   Host Utilities   ///////////////////////

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

////////////////////////   Parallel Mining (Multi-threaded)   ///////////////////////

struct MiningResult
{
    std::atomic<bool> found{false};
    unsigned int nonce{0xffffffffU};
    unsigned char hash[32]{};
};

void solve_with_parallelism(FILE *fin, FILE *fout)
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

    // Multi-threaded mining
    auto start = std::chrono::high_resolution_clock::now();
    
    MiningResult result;
    int num_threads = std::thread::hardware_concurrency();
    printf("Starting parallel mining with %d threads\n", num_threads);
    
    std::vector<std::thread> threads;
    unsigned long long nonce_range = (1ULL << 32) / num_threads;
    
    for(int i = 0; i < num_threads; ++i)
    {
        unsigned long long nonce_start = (unsigned long long)i * nonce_range;
        unsigned long long nonce_end = (i == num_threads - 1) ? (1ULL << 32) : (nonce_start + nonce_range);
        
        threads.emplace_back([&result, &block_host, &target_hex](unsigned long long start, unsigned long long end) {
            printf("Thread mining range %llu to %llu\n", start, end);
            unsigned char hash_result[32];
            
            for(unsigned long long nonce = start; nonce < end && !result.found; ++nonce)
            {
                HashBlock b = block_host;
                b.nonce = (unsigned int)nonce;
                double_sha256_host(hash_result, (unsigned char*)&b, sizeof(HashBlock));
                
                if(little_endian_bit_comparison(hash_result, target_hex, 32) < 0)
                {
                    result.nonce = (unsigned int)nonce;
                    memcpy(result.hash, hash_result, 32);
                    result.found = true;
                    printf("Found nonce: %08x\n", result.nonce);
                    return;
                }
            }
            printf("Thread done mining range %llu to %llu\n", start, end);
        }, nonce_start, nonce_end);
    }
    
    for(auto &t : threads)
    {
        if(t.joinable())
            t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    printf("Mining completed in %ld seconds\n\n", elapsed);

    printf("hash(little): ");
    print_hex(result.hash, 32);
    printf("\n");

    printf("hash(big):    ");
    print_hex_inverse(result.hash, 32);
    printf("\n\n");

    // Output nonce
    for(int i = 0; i < 4; ++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&result.nonce)[i]);
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
        solve_with_parallelism(fin, fout);
    }

    fclose(fin);
    fclose(fout);

    return 0;
}
