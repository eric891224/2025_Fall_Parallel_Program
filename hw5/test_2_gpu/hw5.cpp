#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "hip/hip_runtime.h"

#define HIP_CHECK(call)                                                           \
    do                                                                            \
    {                                                                             \
        hipError_t err = call;                                                    \
        if (err != hipSuccess)                                                    \
        {                                                                         \
            fprintf(stderr, "HIP Error: %s at line %d\n", hipGetErrorString(err), \
                    __LINE__);                                                    \
            exit(1);                                                              \
        }                                                                         \
    } while (0)
#define BLOCK_SIZE 256

const int PLANET_TYPE = 0;
const int DEVICE_TYPE = 1;
const int ASTEROID_TYPE = 2;

namespace param
{
    const int n_steps = 200000;
    const double dt = 60;
    const double eps = 1e-3;
    const double G = 6.674e-11;
    double gravity_device_mass(double m0, double t)
    {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000));
    }
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;
    double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
} // namespace param

__device__ double gravity_device_mass_device(double m0, double t)
{
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}

void read_input(const char *filename, int &n, int &planet, int &asteroid,
                std::vector<double> &qx, std::vector<double> &qy, std::vector<double> &qz,
                std::vector<double> &vx, std::vector<double> &vy, std::vector<double> &vz,
                std::vector<double> &m, std::vector<std::string> &type)
{
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++)
    {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char *filename, double min_dist, int hit_time_step,
                  int gravity_device_id, double missile_cost)
{
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

void run_step(int step, int n, std::vector<double> &qx, std::vector<double> &qy,
              std::vector<double> &qz, std::vector<double> &vx, std::vector<double> &vy,
              std::vector<double> &vz, const std::vector<double> &m,
              const std::vector<std::string> &type)
{
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (j == i)
                continue;
            double mj = m[j];
            if (type[j] == "device")
            {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++)
    {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++)
    {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

__global__ void run_step_kernel(
    int step, int n, int start_idx, int end_idx,
    double dt, double G, double eps,
    const double *__restrict__ qx_in,
    const double *__restrict__ qy_in,
    const double *__restrict__ qz_in,
    double *__restrict__ qx_out,
    double *__restrict__ qy_out,
    double *__restrict__ qz_out,
    double *__restrict__ vx,
    double *__restrict__ vy,
    double *__restrict__ vz,
    const double *__restrict__ m,
    const int *__restrict__ type,
    const double *__restrict__ dev_mass_table,
    const int *__restrict__ dev_index_of_body,
    int num_dev)
{
    extern __shared__ double shmem[];

    int tid = threadIdx.x;
    int i = start_idx + blockIdx.x * blockDim.x + tid;

    // Shared memory layout:
    // s_qx[blockDim.x], s_qy[blockDim.x], s_qz[blockDim.x], s_m[blockDim.x]
    // s_dev_idx[blockDim.x] (as int)
    // s_dev_mass[num_dev] (device masses for this step)
    double *s_qx = shmem;
    double *s_qy = s_qx + blockDim.x;
    double *s_qz = s_qy + blockDim.x;
    double *s_m = s_qz + blockDim.x;
    int *s_dev_idx = (int *)(s_m + blockDim.x);
    double *s_dev_mass = (double *)(s_dev_idx + blockDim.x);

    // Cooperatively load device masses for this step into shared memory
    for (int d = tid; d < num_dev; d += blockDim.x)
    {
        s_dev_mass[d] = dev_mass_table[step * num_dev + d];
    }
    __syncthreads();

    if (i >= end_idx)
        return;

    // Load i-particle from global mem
    double qx_i = qx_in[i];
    double qy_i = qy_in[i];
    double qz_i = qz_in[i];

    double ax = 0.0, ay = 0.0, az = 0.0;

    // --- TILE LOOP ---
    for (int tile = 0; tile < n; tile += blockDim.x)
    {
        int j = tile + tid;

        // Load tile of j-particles into shared memory
        if (j < n)
        {
            s_qx[tid] = qx_in[j];
            s_qy[tid] = qy_in[j];
            s_qz[tid] = qz_in[j];
            s_m[tid] = m[j];
            s_dev_idx[tid] = dev_index_of_body[j];
        }
        __syncthreads();

        int tile_size = min(blockDim.x, n - tile);

        // Interact i-particle with this tile of j-particles
        for (int k = 0; k < tile_size; k++)
        {
            int j_idx = tile + k;
            if (j_idx == i)
                continue;

            double mj = s_m[k];
            // Only use lookup table if mass > 0 (device not destroyed)
            // and it's actually a device (dev_idx >= 0)
            int dev_idx = s_dev_idx[k];
            if (dev_idx >= 0 && mj > 0)
            {
                // lookup from shared memory
                mj = s_dev_mass[dev_idx];
            }

            double dx = s_qx[k] - qx_i;
            double dy = s_qy[k] - qy_i;
            double dz = s_qz[k] - qz_i;

            double invR = rsqrt(dx * dx + dy * dy + dz * dz + eps * eps);
            double invR3 = invR * invR * invR;
            ax += G * mj * dx * invR3;
            ay += G * mj * dy * invR3;
            az += G * mj * dz * invR3;
        }

        __syncthreads();
    }

    // Integrate motion
    vx[i] += ax * dt;
    vy[i] += ay * dt;
    vz[i] += az * dt;

    qx_out[i] = qx_i + vx[i] * dt;
    qy_out[i] = qy_i + vy[i] * dt;
    qz_out[i] = qz_i + vz[i] * dt;
}

__global__ void p1_update_min_dist_kernel(
    int planet, int asteroid,
    const double *qx, const double *qy, const double *qz,
    double *min_dist)
{
    // single-thread kernel
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        double p_qx = qx[planet];
        double p_qy = qy[planet];
        double p_qz = qz[planet];
        double a_qx = qx[asteroid];
        double a_qy = qy[asteroid];
        double a_qz = qz[asteroid];

        double dx = p_qx - a_qx;
        double dy = p_qy - a_qy;
        double dz = p_qz - a_qz;
        double dist = sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < *min_dist)
        {
            *min_dist = dist;
        }
    }
}

__global__ void p2_check_hit_kernel(
    int step, int planet, int asteroid,
    const double *qx, const double *qy, const double *qz,
    int *hit_step)
{
    // single-thread kernel
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        double p_qx = qx[planet];
        double p_qy = qy[planet];
        double p_qz = qz[planet];
        double a_qx = qx[asteroid];
        double a_qy = qy[asteroid];
        double a_qz = qz[asteroid];

        double dx = p_qx - a_qx;
        double dy = p_qy - a_qy;
        double dz = p_qz - a_qz;
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius)
        {
            // only write if unset
            if (*hit_step < 0)
            {
                *hit_step = step;
            }
        }
    }
}

__global__ void p3_distance_kernel(
    int planet, int asteroid,
    const double *qx, const double *qy, const double *qz,
    const int *device_indices, int num_dev,
    double *d_ast_dist, // single double
    double *d_dev_dist  // array [num_dev]
)
{
    int tid = threadIdx.x;

    // asteroid–planet distance (thread 0)
    if (tid == 0)
    {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        *d_ast_dist = sqrt(dx * dx + dy * dy + dz * dz);
    }

    // planet–device distances
    if (tid < num_dev)
    {
        int dev = device_indices[tid];
        double dx = qx[planet] - qx[dev];
        double dy = qy[planet] - qy[dev];
        double dz = qz[planet] - qz[dev];
        d_dev_dist[tid] = sqrt(dx * dx + dy * dy + dz * dz);
    }
}

void solve_p1(
    int n, int planet, int asteroid,
    std::vector<double> &h_qx, std::vector<double> &h_qy, std::vector<double> &h_qz,
    std::vector<double> &h_vx, std::vector<double> &h_vy, std::vector<double> &h_vz,
    std::vector<double> &h_m, std::vector<int> &h_type,
    double &min_dist,
    int gpu_id = 0)
{
    // Device pointers
    double *d_qx[2], *d_qy[2], *d_qz[2]; // [buffer_idx]
    double *d_vx, *d_vy, *d_vz;
    double *d_m;
    int *d_type;

    // p1 specific
    double *d_min_dist;

    // Use specified GPU
    HIP_CHECK(hipSetDevice(gpu_id));

    HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double)));
    HIP_CHECK(hipMemcpy(d_min_dist, &min_dist, sizeof(double), hipMemcpyHostToDevice));

    // allocate positions with double buffering
    for (int buf = 0; buf < 2; buf++)
    {
        HIP_CHECK(hipMalloc(&d_qx[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz[buf], n * sizeof(double)));
    }
    // allocate velocities
    HIP_CHECK(hipMalloc(&d_vx, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vy, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vz, n * sizeof(double)));
    // allocate masses
    HIP_CHECK(hipMalloc(&d_m, n * sizeof(double)));
    // allocate types
    HIP_CHECK(hipMalloc(&d_type, n * sizeof(int)));
    // H2D copies
    HIP_CHECK(hipMemcpy(d_qx[0], h_qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy[0], h_qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz[0], h_qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, h_vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, h_vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, h_vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, h_m.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_type, h_type.data(), n * sizeof(int), hipMemcpyHostToDevice));

    // Precompute device mass table
    int num_dev = 0;
    std::vector<int> dev_index_of_body(n, -1);
    std::vector<int> device_indices_p1;
    for (int i = 0; i < n; i++)
    {
        if (h_type[i] == DEVICE_TYPE)
        {
            dev_index_of_body[i] = num_dev++;
            device_indices_p1.push_back(i);
        }
    }

    // Transposed layout: table[step * num_dev + d] for coalesced access
    std::vector<double> host_dev_mass_table(num_dev * param::n_steps);
    for (int step = 0; step < param::n_steps; step++)
    {
        double t = step * param::dt;
        for (int d = 0; d < num_dev; d++)
        {
            double m0 = h_m[device_indices_p1[d]];
            host_dev_mass_table[step * num_dev + d] =
                param::gravity_device_mass(m0, t);
        }
    }

    double *d_dev_mass_table;
    int *d_dev_index_of_body;
    size_t table_size = host_dev_mass_table.size() * sizeof(double);
    if (table_size == 0)
        table_size = sizeof(double); // avoid zero allocation
    HIP_CHECK(hipMalloc(&d_dev_mass_table, table_size));
    if (num_dev > 0)
    {
        HIP_CHECK(hipMemcpy(d_dev_mass_table,
                            host_dev_mass_table.data(),
                            host_dev_mass_table.size() * sizeof(double),
                            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMalloc(&d_dev_index_of_body, n * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_dev_index_of_body,
                        dev_index_of_body.data(),
                        n * sizeof(int),
                        hipMemcpyHostToDevice));

    int cur = 0;
    int next = 1;
    int blockSize = BLOCK_SIZE;

    for (int step = 1; step < param::n_steps; step++)
    {
        int gridSize = (n + blockSize - 1) / blockSize;
        size_t shmem_size = 4 * blockSize * sizeof(double) + blockSize * sizeof(int) + num_dev * sizeof(double);

        run_step_kernel<<<gridSize, blockSize, shmem_size>>>(
            step, n,
            0, n,
            param::dt, param::G, param::eps,
            d_qx[cur], d_qy[cur], d_qz[cur],
            d_qx[next], d_qy[next], d_qz[next],
            d_vx, d_vy, d_vz,
            d_m, d_type,
            d_dev_mass_table,
            d_dev_index_of_body,
            num_dev);
        HIP_CHECK(hipGetLastError());

        // update min_dist, TODO: use kernel
        p1_update_min_dist_kernel<<<1, 1>>>(planet, asteroid,
                                            d_qx[next], d_qy[next], d_qz[next],
                                            d_min_dist);
        // double p_qx, p_qy, p_qz;
        // double a_qx, a_qy, a_qz;
        // HIP_CHECK(hipMemcpy(&p_qx, d_qx[next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        // HIP_CHECK(hipMemcpy(&p_qy, d_qy[next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        // HIP_CHECK(hipMemcpy(&p_qz, d_qz[next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        // HIP_CHECK(hipMemcpy(&a_qx, d_qx[next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        // HIP_CHECK(hipMemcpy(&a_qy, d_qy[next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        // HIP_CHECK(hipMemcpy(&a_qz, d_qz[next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        // double dx = p_qx - a_qx;
        // double dy = p_qy - a_qy;
        // double dz = p_qz - a_qz;
        // double dist = sqrt(dx * dx + dy * dy + dz * dz);
        // if (dist < min_dist)
        //     min_dist = dist;

        // Swap buffers for double buffering
        std::swap(cur, next);
    }

    HIP_CHECK(hipMemcpy(&min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));

    // clean up
    for (int buf = 0; buf < 2; buf++)
    {
        HIP_CHECK(hipFree(d_qx[buf]));
        HIP_CHECK(hipFree(d_qy[buf]));
        HIP_CHECK(hipFree(d_qz[buf]));
    }
    HIP_CHECK(hipFree(d_vx));
    HIP_CHECK(hipFree(d_vy));
    HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_type));
    HIP_CHECK(hipFree(d_min_dist));
    HIP_CHECK(hipFree(d_dev_mass_table));
    HIP_CHECK(hipFree(d_dev_index_of_body));
}

void solve_p2(
    int n, int planet, int asteroid,
    std::vector<double> &h_qx, std::vector<double> &h_qy, std::vector<double> &h_qz,
    std::vector<double> &h_vx, std::vector<double> &h_vy, std::vector<double> &h_vz,
    std::vector<double> &h_m, std::vector<int> &h_type,
    int &hit_time_step,
    int start_step = 1,
    int gpu_id = 0)
{
    // Device pointers
    double *d_qx[2], *d_qy[2], *d_qz[2]; // [buffer_idx]
    double *d_vx, *d_vy, *d_vz;
    double *d_m;
    int *d_type;

    // Use specified GPU
    HIP_CHECK(hipSetDevice(gpu_id));

    // problem 2 specific
    int *d_hit_step;
    HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_hit_step, &hit_time_step, sizeof(int), hipMemcpyHostToDevice));

    // allocate positions with double buffering
    for (int buf = 0; buf < 2; buf++)
    {
        HIP_CHECK(hipMalloc(&d_qx[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz[buf], n * sizeof(double)));
    }
    // allocate velocities
    HIP_CHECK(hipMalloc(&d_vx, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vy, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vz, n * sizeof(double)));
    // allocate masses
    HIP_CHECK(hipMalloc(&d_m, n * sizeof(double)));
    // allocate types
    HIP_CHECK(hipMalloc(&d_type, n * sizeof(int)));
    // H2D copies
    HIP_CHECK(hipMemcpy(d_qx[0], h_qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy[0], h_qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz[0], h_qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, h_vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, h_vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, h_vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, h_m.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_type, h_type.data(), n * sizeof(int), hipMemcpyHostToDevice));

    // Precompute device mass table
    int num_dev = 0;
    std::vector<int> dev_index_of_body(n, -1);
    std::vector<int> device_indices_p2;
    for (int i = 0; i < n; i++)
    {
        if (h_type[i] == DEVICE_TYPE)
        {
            dev_index_of_body[i] = num_dev++;
            device_indices_p2.push_back(i);
        }
    }

    // Transposed layout: table[step * num_dev + d] for coalesced access
    std::vector<double> host_dev_mass_table(num_dev * param::n_steps);
    for (int step = 0; step < param::n_steps; step++)
    {
        double t = step * param::dt;
        for (int d = 0; d < num_dev; d++)
        {
            double m0 = h_m[device_indices_p2[d]];
            host_dev_mass_table[step * num_dev + d] =
                param::gravity_device_mass(m0, t);
        }
    }

    double *d_dev_mass_table;
    int *d_dev_index_of_body;
    size_t table_size = host_dev_mass_table.size() * sizeof(double);
    if (table_size == 0)
        table_size = sizeof(double); // avoid zero allocation
    HIP_CHECK(hipMalloc(&d_dev_mass_table, table_size));
    if (num_dev > 0)
    {
        HIP_CHECK(hipMemcpy(d_dev_mass_table,
                            host_dev_mass_table.data(),
                            host_dev_mass_table.size() * sizeof(double),
                            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMalloc(&d_dev_index_of_body, n * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_dev_index_of_body,
                        dev_index_of_body.data(),
                        n * sizeof(int),
                        hipMemcpyHostToDevice));

    int cur = 0;
    int next = 1;
    int blockSize = BLOCK_SIZE;

    for (int step = start_step; step < param::n_steps; step++)
    {
        // launch kernel
        int gridSize = (n + blockSize - 1) / blockSize;
        size_t shmem_size = 4 * blockSize * sizeof(double) + blockSize * sizeof(int) + num_dev * sizeof(double);

        run_step_kernel<<<gridSize, blockSize, shmem_size>>>(
            step, n,
            0, n,
            param::dt, param::G, param::eps,
            d_qx[cur], d_qy[cur], d_qz[cur],
            d_qx[next], d_qy[next], d_qz[next],
            d_vx, d_vy, d_vz,
            d_m, d_type,
            d_dev_mass_table,
            d_dev_index_of_body,
            num_dev);
        HIP_CHECK(hipGetLastError());

        // check if hit
        p2_check_hit_kernel<<<1, 1>>>(step, planet, asteroid, d_qx[next], d_qy[next], d_qz[next], d_hit_step);

        // Swap buffers for double buffering
        std::swap(cur, next);
    }

    HIP_CHECK(hipMemcpy(&hit_time_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));

    // clean up
    for (int buf = 0; buf < 2; buf++)
    {
        HIP_CHECK(hipFree(d_qx[buf]));
        HIP_CHECK(hipFree(d_qy[buf]));
        HIP_CHECK(hipFree(d_qz[buf]));
    }
    HIP_CHECK(hipFree(d_vx));
    HIP_CHECK(hipFree(d_vy));
    HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_type));
    HIP_CHECK(hipFree(d_hit_step));
    HIP_CHECK(hipFree(d_dev_mass_table));
    HIP_CHECK(hipFree(d_dev_index_of_body));
}

void copy_state(
    std::vector<double> &temp_qx, std::vector<double> &temp_qy, std::vector<double> &temp_qz,
    std::vector<double> &temp_vx, std::vector<double> &temp_vy, std::vector<double> &temp_vz,
    std::vector<double> &temp_m,
    const std::vector<double> &qx, const std::vector<double> &qy, const std::vector<double> &qz,
    const std::vector<double> &vx, const std::vector<double> &vy, const std::vector<double> &vz,
    const std::vector<double> &m)
{
    temp_qx = qx;
    temp_qy = qy;
    temp_qz = qz;
    temp_vx = vx;
    temp_vy = vy;
    temp_vz = vz;
    temp_m = m;
}

void solve_p3(
    int n, int planet, int asteroid,
    std::vector<double> &h_qx, std::vector<double> &h_qy, std::vector<double> &h_qz,
    std::vector<double> &h_vx, std::vector<double> &h_vy, std::vector<double> &h_vz,
    std::vector<double> &h_m, std::vector<int> &h_type,
    int &gravity_device_id, double &missile_cost, int &hit_time_step, int gpu_device = 0)
{
    HIP_CHECK(hipSetDevice(gpu_device));

    // collect gravity-device indices on host
    std::vector<int> device_indices;
    for (int i = 0; i < n; i++)
    {
        if (h_type[i] == DEVICE_TYPE)
        {
            device_indices.push_back(i);
        }
    }

    // Device pointers (main simulation)
    double *d_qx[2], *d_qy[2], *d_qz[2]; // ping-pong
    double *d_vx, *d_vy, *d_vz;
    double *d_m;
    int *d_type;

    // allocate positions with double buffering
    for (int buf = 0; buf < 2; buf++)
    {
        HIP_CHECK(hipMalloc(&d_qx[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz[buf], n * sizeof(double)));
    }
    // allocate velocities
    HIP_CHECK(hipMalloc(&d_vx, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vy, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vz, n * sizeof(double)));
    // allocate masses
    HIP_CHECK(hipMalloc(&d_m, n * sizeof(double)));
    // allocate types
    HIP_CHECK(hipMalloc(&d_type, n * sizeof(int)));

    // H2D copies (initial state)
    HIP_CHECK(hipMemcpy(d_qx[0], h_qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy[0], h_qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz[0], h_qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, h_vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, h_vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, h_vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, h_m.data(), n * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_type, h_type.data(), n * sizeof(int), hipMemcpyHostToDevice));

    // Precompute device mass table
    int num_dev_p3 = 0;
    std::vector<int> dev_index_of_body(n, -1);
    std::vector<int> device_indices_for_table;
    for (int i = 0; i < n; i++)
    {
        if (h_type[i] == DEVICE_TYPE)
        {
            dev_index_of_body[i] = num_dev_p3++;
            device_indices_for_table.push_back(i);
        }
    }

    // Transposed layout: table[step * num_dev + d] for coalesced access
    std::vector<double> host_dev_mass_table(num_dev_p3 * param::n_steps);
    for (int step = 0; step < param::n_steps; step++)
    {
        double t = step * param::dt;
        for (int d = 0; d < num_dev_p3; d++)
        {
            double m0 = h_m[device_indices_for_table[d]];
            host_dev_mass_table[step * num_dev_p3 + d] =
                param::gravity_device_mass(m0, t);
        }
    }

    double *d_dev_mass_table;
    int *d_dev_index_of_body;
    size_t table_size = host_dev_mass_table.size() * sizeof(double);
    if (table_size == 0)
        table_size = sizeof(double); // avoid zero allocation
    HIP_CHECK(hipMalloc(&d_dev_mass_table, table_size));
    if (num_dev_p3 > 0)
    {
        HIP_CHECK(hipMemcpy(d_dev_mass_table,
                            host_dev_mass_table.data(),
                            host_dev_mass_table.size() * sizeof(double),
                            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMalloc(&d_dev_index_of_body, n * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_dev_index_of_body,
                        dev_index_of_body.data(),
                        n * sizeof(int),
                        hipMemcpyHostToDevice));

    // --- extra device buffers for P3 logic ---

    // backup (snapshot) of state: ping-pong + v + m
    double *d_qx_backup[2], *d_qy_backup[2], *d_qz_backup[2];
    double *d_vx_backup, *d_vy_backup, *d_vz_backup;
    double *d_m_backup;

    for (int buf = 0; buf < 2; buf++)
    {
        HIP_CHECK(hipMalloc(&d_qx_backup[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy_backup[buf], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz_backup[buf], n * sizeof(double)));
    }
    HIP_CHECK(hipMalloc(&d_vx_backup, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vy_backup, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vz_backup, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_m_backup, n * sizeof(double)));

    // device data for distance checks
    int max_devices = static_cast<int>(device_indices.size());
    if (max_devices == 0)
    {
        // no gravity devices => cannot prevent hit
        gravity_device_id = -1;
        missile_cost = 0.0;

        // free main device buffers
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipFree(d_qx[buf]));
            HIP_CHECK(hipFree(d_qy[buf]));
            HIP_CHECK(hipFree(d_qz[buf]));
            HIP_CHECK(hipFree(d_qx_backup[buf]));
            HIP_CHECK(hipFree(d_qy_backup[buf]));
            HIP_CHECK(hipFree(d_qz_backup[buf]));
        }
        HIP_CHECK(hipFree(d_vx));
        HIP_CHECK(hipFree(d_vy));
        HIP_CHECK(hipFree(d_vz));
        HIP_CHECK(hipFree(d_m));
        HIP_CHECK(hipFree(d_type));
        HIP_CHECK(hipFree(d_vx_backup));
        HIP_CHECK(hipFree(d_vy_backup));
        HIP_CHECK(hipFree(d_vz_backup));
        HIP_CHECK(hipFree(d_m_backup));
        HIP_CHECK(hipFree(d_dev_mass_table));
        HIP_CHECK(hipFree(d_dev_index_of_body));
        return;
    }

    int *d_device_indices;
    double *d_dev_dist; // distances from planet to each device
    double *d_ast_dist; // asteroid–planet distance

    HIP_CHECK(hipMalloc(&d_device_indices, max_devices * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_dev_dist, max_devices * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_ast_dist, sizeof(double)));

    // for branch simulation: reuse P2's hit_step mechanism
    int *d_hit_step;
    HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));

    // host scratch for distances
    std::vector<double> h_dev_dist(max_devices);
    double h_ast_dist;

    int cur = 0;
    int next = 1;
    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t shmem_size = 4 * blockSize * sizeof(double) + blockSize * sizeof(int) + num_dev_p3 * sizeof(double);

    // initialize outputs
    gravity_device_id = -999;
    missile_cost = -999;
    hit_time_step = -2; // Initialize: no hit yet

    // host lambda: snapshot current device state (after a step)
    auto snapshot_state = [&]()
    {
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipMemcpy(d_qx_backup[buf], d_qx[buf], n * sizeof(double), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(d_qy_backup[buf], d_qy[buf], n * sizeof(double), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(d_qz_backup[buf], d_qz[buf], n * sizeof(double), hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipMemcpy(d_vx_backup, d_vx, n * sizeof(double), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_vy_backup, d_vy, n * sizeof(double), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_vz_backup, d_vz, n * sizeof(double), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_m_backup, d_m, n * sizeof(double), hipMemcpyDeviceToDevice));
    };

    // host lambda: restore state from snapshot
    auto restore_state = [&]()
    {
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipMemcpy(d_qx[buf], d_qx_backup[buf], n * sizeof(double), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(d_qy[buf], d_qy_backup[buf], n * sizeof(double), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(d_qz[buf], d_qz_backup[buf], n * sizeof(double), hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipMemcpy(d_vx, d_vx_backup, n * sizeof(double), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_vy, d_vy_backup, n * sizeof(double), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_vz, d_vz_backup, n * sizeof(double), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_m, d_m_backup, n * sizeof(double), hipMemcpyDeviceToDevice));
    };

    // branch simulation: starting from state at "next" buffer, with one device mass set to 0
    auto branch_still_hits = [&](int destroy_device_id, int start_step) -> bool
    {
        // restore base state at this step
        restore_state();

        // set device mass to 0 (double)
        double zero = 0.0;
        HIP_CHECK(hipMemcpy(d_m + destroy_device_id, &zero, sizeof(double), hipMemcpyHostToDevice));

        // hit_step init on device
        int hit_init = -2;
        HIP_CHECK(hipMemcpy(d_hit_step, &hit_init, sizeof(int), hipMemcpyHostToDevice));

        // branch uses same ping-pong buffers, starting from "next"
        int b_cur = next;
        int b_next = cur;

        // Check hit every CHECK_INTERVAL steps to reduce sync overhead
        const int CHECK_INTERVAL = 5000;

        for (int s = start_step; s < param::n_steps; s++)
        {
            run_step_kernel<<<gridSize, blockSize, shmem_size>>>(
                s, n,
                0, n,
                param::dt, param::G, param::eps,
                d_qx[b_cur], d_qy[b_cur], d_qz[b_cur],
                d_qx[b_next], d_qy[b_next], d_qz[b_next],
                d_vx, d_vy, d_vz,
                d_m, d_type,
                d_dev_mass_table,
                d_dev_index_of_body,
                num_dev_p3);

            p2_check_hit_kernel<<<1, 1>>>(
                s, planet, asteroid,
                d_qx[b_next], d_qy[b_next], d_qz[b_next],
                d_hit_step);

            // Only check periodically to reduce host-device sync
            if ((s - start_step + 1) % CHECK_INTERVAL == 0)
            {
                int host_hit_step;
                HIP_CHECK(hipMemcpy(&host_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
                if (host_hit_step >= 0)
                {
                    return true;
                }
            }

            std::swap(b_cur, b_next);
        }

        // Final check
        HIP_CHECK(hipDeviceSynchronize());
        int host_hit_step;
        HIP_CHECK(hipMemcpy(&host_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
        return host_hit_step >= 0;
    };

    // --- main P3 simulation loop (GPU only, minimal host traffic) ---

    // Upload device_indices initially (only re-upload when list changes)
    bool device_indices_dirty = true;

    for (int step = 1; step < param::n_steps; step++)
    {
        // 1. advance one step (cur -> next)
        run_step_kernel<<<gridSize, blockSize, shmem_size>>>(
            step, n,
            0, n,
            param::dt, param::G, param::eps,
            d_qx[cur], d_qy[cur], d_qz[cur],
            d_qx[next], d_qy[next], d_qz[next],
            d_vx, d_vy, d_vz,
            d_m, d_type,
            d_dev_mass_table,
            d_dev_index_of_body,
            num_dev_p3);

        // if no more devices to consider, we can early out after checking if asteroid hits
        int num_dev = static_cast<int>(device_indices.size());

        // 2. distance checks for this step (using positions in "next")
        // Only upload device_indices when the list has changed
        if (num_dev > 0 && device_indices_dirty)
        {
            HIP_CHECK(hipMemcpy(d_device_indices,
                                device_indices.data(),
                                num_dev * sizeof(int),
                                hipMemcpyHostToDevice));
            device_indices_dirty = false;
        }

        int threads = (num_dev > 0) ? num_dev : 1;
        p3_distance_kernel<<<1, threads>>>(
            planet, asteroid,
            d_qx[next], d_qy[next], d_qz[next],
            d_device_indices, num_dev,
            d_ast_dist, d_dev_dist);

        HIP_CHECK(hipMemcpy(&h_ast_dist, d_ast_dist, sizeof(double), hipMemcpyDeviceToHost));

        // 3. asteroid hits planet in the "no missile" baseline
        if (h_ast_dist * h_ast_dist < param::planet_radius * param::planet_radius)
        {
            // Record hit time for P2 functionality
            hit_time_step = step;
            // if no device ever prevented a hit, same semantics as original:
            if (missile_cost == -999)
            {
                gravity_device_id = -1;
                missile_cost = 0.0;
            }
            break;
        }

        // 4. if there are devices, get their distances
        if (num_dev > 0)
        {
            HIP_CHECK(hipMemcpy(h_dev_dist.data(), d_dev_dist,
                                num_dev * sizeof(double),
                                hipMemcpyDeviceToHost));
        }

        // missile distance from planet at this step
        double missile_dist = step * param::dt * param::missile_speed;

        // we may test multiple devices this step; snapshot state once if needed
        bool snap_taken = false;

        // we'll build a list of indices in device_indices that get hit this step
        std::vector<int> hit_indices;

        for (int idx = 0; idx < num_dev; idx++)
        {
            if (missile_dist > h_dev_dist[idx])
            {
                hit_indices.push_back(idx);
            }
        }

        if (!hit_indices.empty())
        {
            // snapshot the current state (after step, before swap)
            snapshot_state();

            // for each candidate device destroyed at this step, simulate branch
            for (int local_idx : hit_indices)
            {
                int device_id = device_indices[local_idx];

                bool still_hits = branch_still_hits(device_id, step + 1);

                if (!still_hits)
                {
                    // asteroid is saved by destroying this device now
                    double cost = param::get_missile_cost((step + 1) * param::dt);
                    if (missile_cost == -999 || cost < missile_cost)
                    {
                        missile_cost = cost;
                        gravity_device_id = device_id;
                    }
                }
            }

            // Restore state after all branches so main simulation can continue
            restore_state();

            // now remove all devices that have been destroyed by the missile so far
            // (those with index in hit_indices)
            // remove from highest index to lowest to keep indices valid
            std::sort(hit_indices.begin(), hit_indices.end(), std::greater<int>());
            for (int idx : hit_indices)
            {
                device_indices.erase(device_indices.begin() + idx);
            }
            // Mark device_indices as dirty so it gets re-uploaded
            device_indices_dirty = true;
        }

        // 5. swap ping-pong buffers for main simulation
        std::swap(cur, next);

        // optional small optimization: if no devices remain and we've already found
        // some valid solution, we *could* stop once the asteroid baseline hits,
        // but we already break in that case above.
    }

    // --- cleanup ---

    for (int buf = 0; buf < 2; buf++)
    {
        HIP_CHECK(hipFree(d_qx[buf]));
        HIP_CHECK(hipFree(d_qy[buf]));
        HIP_CHECK(hipFree(d_qz[buf]));
        HIP_CHECK(hipFree(d_qx_backup[buf]));
        HIP_CHECK(hipFree(d_qy_backup[buf]));
        HIP_CHECK(hipFree(d_qz_backup[buf]));
    }
    HIP_CHECK(hipFree(d_vx));
    HIP_CHECK(hipFree(d_vy));
    HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_type));
    HIP_CHECK(hipFree(d_vx_backup));
    HIP_CHECK(hipFree(d_vy_backup));
    HIP_CHECK(hipFree(d_vz_backup));
    HIP_CHECK(hipFree(d_m_backup));

    HIP_CHECK(hipFree(d_device_indices));
    HIP_CHECK(hipFree(d_dev_dist));
    HIP_CHECK(hipFree(d_ast_dist));
    HIP_CHECK(hipFree(d_hit_step));
    HIP_CHECK(hipFree(d_dev_mass_table));
    HIP_CHECK(hipFree(d_dev_index_of_body));
}

void convert_type(const std::vector<std::string> &type_str, std::vector<int> &type_int)
{
    type_int.resize(type_str.size());
    for (int i = 0; i < type_str.size(); i++)
    {
        if (type_str[i] == "device")
        {
            type_int[i] = DEVICE_TYPE;
        }
        else if (type_str[i] == "planet")
        {
            type_int[i] = PLANET_TYPE;
        }
        else if (type_str[i] == "asteroid")
        {
            type_int[i] = ASTEROID_TYPE;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        throw std::runtime_error("must supply 2 arguments");
    }

    using clock = std::chrono::high_resolution_clock;

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    std::vector<int> type_int;

    // Read input once
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    convert_type(type, type_int);

    // Make copies for each problem (since they modify the data)
    std::vector<double> qx1 = qx, qy1 = qy, qz1 = qz, vx1 = vx, vy1 = vy, vz1 = vz, m1 = m;
    std::vector<int> type_int1 = type_int;
    std::vector<double> qx2 = qx, qy2 = qy, qz2 = qz, vx2 = vx, vy2 = vy, vz2 = vz, m2 = m;
    std::vector<int> type_int2 = type_int;

    // Zero out device masses for P1
    for (int i = 0; i < n; i++)
    {
        if (type_int1[i] == DEVICE_TYPE)
        {
            m1[i] = 0;
        }
    }

    // Run P1 and P3 in parallel on 2 GPUs
    auto t1 = clock::now();
    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    int gravity_device_id = -999;
    double missile_cost = -999;

    // Launch P1 on GPU 0 and P3 on GPU 1 concurrently
    std::thread thread_p1([&]()
                          { solve_p1(n, planet, asteroid, qx1, qy1, qz1, vx1, vy1, vz1, m1, type_int1, min_dist, 0); });

    std::thread thread_p3([&]()
                          { solve_p3(n, planet, asteroid, qx2, qy2, qz2, vx2, vy2, vz2, m2, type_int2,
                                     gravity_device_id, missile_cost, hit_time_step, 1); });

    // Wait for both threads to complete
    thread_p1.join();
    thread_p3.join();

    auto t2 = clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "Problem 1+3 elapsed time (2 GPUs): " << elapsed.count() << " seconds\n";

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
