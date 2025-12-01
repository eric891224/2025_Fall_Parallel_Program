#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
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
    double dt, double G, double eps, // these are params
    const double *__restrict__ qx_in, const double *__restrict__ qy_in, const double *__restrict__ qz_in,
    double *__restrict__ qx_out, double *__restrict__ qy_out, double *__restrict__ qz_out,
    double *__restrict__ vx, double *__restrict__ vy, double *__restrict__ vz,
    const double *__restrict__ m, const int *__restrict__ type)
{
    int i = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end_idx)
        return;

    // compute accelerations
    double ax = 0.0, ay = 0.0, az = 0.0;
    double qx_i = qx_in[i];
    double qy_i = qy_in[i];
    double qz_i = qz_in[i];
    for (int j = 0; j < n; j++)
    {
        if (j == i)
            continue;

        double mj = m[j];
        if (type[j] == DEVICE_TYPE && mj > 0) // branch divergence
        {
            mj = gravity_device_mass_device(mj, step * dt);
        }

        double dx = qx_in[j] - qx_i;
        double dy = qy_in[j] - qy_i;
        double dz = qz_in[j] - qz_i;
        double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);
        ax += G * mj * dx / dist3;
        ay += G * mj * dy / dist3;
        az += G * mj * dz / dist3;
    }
    // update velocities
    vx[i] += ax * dt;
    vy[i] += ay * dt;
    vz[i] += az * dt;

    // update positions
    qx_out[i] = qx_i + vx[i] * dt;
    qy_out[i] = qy_i + vy[i] * dt;
    qz_out[i] = qz_i + vz[i] * dt;
}

void solve_p1(
    int n, int planet, int asteroid,
    std::vector<double> &h_qx, std::vector<double> &h_qy, std::vector<double> &h_qz,
    std::vector<double> &h_vx, std::vector<double> &h_vy, std::vector<double> &h_vz,
    std::vector<double> &h_m, std::vector<int> &h_type,
    double &min_dist)
{
    // Split workload
    int mid = n / 2;
    int ranges[2][2] = {{0, mid}, {mid, n}};
    int counts[2] = {mid, n - mid};

    // Device pointers
    double *d_qx[2][2], *d_qy[2][2], *d_qz[2][2]; // [gpu][buffer_idx]
    double *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_m[2];
    int *d_type[2];

    // Allocate device memory and copy data H2D
    for (int gpu = 0; gpu < 2; gpu++)
    {
        HIP_CHECK(hipSetDevice(gpu));

        // allocate positions with double buffering
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipMalloc(&d_qx[gpu][buf], n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qy[gpu][buf], n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qz[gpu][buf], n * sizeof(double)));
        }
        // allocate velocities
        HIP_CHECK(hipMalloc(&d_vx[gpu], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy[gpu], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz[gpu], n * sizeof(double)));
        // allocate masses
        HIP_CHECK(hipMalloc(&d_m[gpu], n * sizeof(double)));
        // allocate types
        HIP_CHECK(hipMalloc(&d_type[gpu], n * sizeof(int)));

        // H2D copies
        HIP_CHECK(hipMemcpy(d_qx[gpu][0], h_qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[gpu][0], h_qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[gpu][0], h_qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx[gpu], h_vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy[gpu], h_vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz[gpu], h_vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m[gpu], h_m.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_type[gpu], h_type.data(), n * sizeof(int), hipMemcpyHostToDevice));
    }

    int cur = 0;
    int next = 1;
    int blockSize = 256;

    for (int step = 1; step < param::n_steps; step++)
    {
        // Launch kernels on both GPUs
        for (int gpu = 0; gpu < 2; gpu++)
        {
            HIP_CHECK(hipSetDevice(gpu));
            int count = counts[gpu];
            int gridSize = (count + blockSize - 1) / blockSize;

            run_step_kernel<<<gridSize, blockSize>>>(
                step, n,
                ranges[gpu][0], ranges[gpu][1],
                param::dt, param::G, param::eps,
                d_qx[gpu][cur], d_qy[gpu][cur], d_qz[gpu][cur],
                d_qx[gpu][next], d_qy[gpu][next], d_qz[gpu][next],
                d_vx[gpu], d_vy[gpu], d_vz[gpu],
                d_m[gpu], d_type[gpu]);
            HIP_CHECK(hipGetLastError());
        }

        // synchronize
        for (int gpu = 0; gpu < 2; gpu++)
        {
            HIP_CHECK(hipSetDevice(gpu));
            HIP_CHECK(hipDeviceSynchronize());
        }

        // exchange data
        // GPU0 [0, mid) -> GPU1
        // GPU1 [mid, n) -> GPU0
        HIP_CHECK(hipMemcpyPeer(d_qx[1][next], 1, d_qx[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qy[1][next], 1, d_qy[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qz[1][next], 1, d_qz[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qx[0][next] + mid, 0, d_qx[1][next] + mid, 1, (n - mid) * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qy[0][next] + mid, 0, d_qy[1][next] + mid, 1, (n - mid) * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qz[0][next] + mid, 0, d_qz[1][next] + mid, 1, (n - mid) * sizeof(double)));

        // update min_dist
        double p_qx, p_qy, p_qz;
        double a_qx, a_qy, a_qz;
        HIP_CHECK(hipMemcpy(&p_qx, d_qx[0][next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&p_qy, d_qy[0][next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&p_qz, d_qz[0][next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&a_qx, d_qx[0][next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&a_qy, d_qy[0][next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&a_qz, d_qz[0][next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        double dx = p_qx - a_qx;
        double dy = p_qy - a_qy;
        double dz = p_qz - a_qz;
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < min_dist)
            min_dist = dist;

        // Swap buffers for double buffering
        std::swap(cur, next);
    }

    // clean up
    for (int gpu = 0; gpu < 2; gpu++)
    {
        HIP_CHECK(hipSetDevice(gpu));
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipFree(d_qx[gpu][buf]));
            HIP_CHECK(hipFree(d_qy[gpu][buf]));
            HIP_CHECK(hipFree(d_qz[gpu][buf]));
        }
        HIP_CHECK(hipFree(d_vx[gpu]));
        HIP_CHECK(hipFree(d_vy[gpu]));
        HIP_CHECK(hipFree(d_vz[gpu]));
        HIP_CHECK(hipFree(d_m[gpu]));
        HIP_CHECK(hipFree(d_type[gpu]));
    }
}

void solve_p2(
    int n, int planet, int asteroid,
    std::vector<double> &h_qx, std::vector<double> &h_qy, std::vector<double> &h_qz,
    std::vector<double> &h_vx, std::vector<double> &h_vy, std::vector<double> &h_vz,
    std::vector<double> &h_m, std::vector<int> &h_type,
    int &hit_time_step,
    int start_step = 1)
{
    // Split workload
    int mid = n / 2;
    int ranges[2][2] = {{0, mid}, {mid, n}};
    int counts[2] = {mid, n - mid};

    // Device pointers
    double *d_qx[2][2], *d_qy[2][2], *d_qz[2][2]; // [gpu][buffer_idx]
    double *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_m[2];
    int *d_type[2];

    // Allocate device memory and copy data H2D
    for (int gpu = 0; gpu < 2; gpu++)
    {
        HIP_CHECK(hipSetDevice(gpu));

        // allocate positions with double buffering
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipMalloc(&d_qx[gpu][buf], n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qy[gpu][buf], n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qz[gpu][buf], n * sizeof(double)));
        }
        // allocate velocities
        HIP_CHECK(hipMalloc(&d_vx[gpu], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy[gpu], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz[gpu], n * sizeof(double)));
        // allocate masses
        HIP_CHECK(hipMalloc(&d_m[gpu], n * sizeof(double)));
        // allocate types
        HIP_CHECK(hipMalloc(&d_type[gpu], n * sizeof(int)));

        // H2D copies
        HIP_CHECK(hipMemcpy(d_qx[gpu][0], h_qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[gpu][0], h_qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[gpu][0], h_qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx[gpu], h_vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy[gpu], h_vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz[gpu], h_vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m[gpu], h_m.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_type[gpu], h_type.data(), n * sizeof(int), hipMemcpyHostToDevice));
    }

    int cur = 0;
    int next = 1;
    int blockSize = 256;

    for (int step = start_step; step < param::n_steps; step++)
    {
        // Launch kernels on both GPUs
        for (int gpu = 0; gpu < 2; gpu++)
        {
            HIP_CHECK(hipSetDevice(gpu));
            int count = counts[gpu];
            int gridSize = (count + blockSize - 1) / blockSize;

            run_step_kernel<<<gridSize, blockSize>>>(
                step, n,
                ranges[gpu][0], ranges[gpu][1],
                param::dt, param::G, param::eps,
                d_qx[gpu][cur], d_qy[gpu][cur], d_qz[gpu][cur],
                d_qx[gpu][next], d_qy[gpu][next], d_qz[gpu][next],
                d_vx[gpu], d_vy[gpu], d_vz[gpu],
                d_m[gpu], d_type[gpu]);
            HIP_CHECK(hipGetLastError());
        }

        // synchronize
        for (int gpu = 0; gpu < 2; gpu++)
        {
            HIP_CHECK(hipSetDevice(gpu));
            HIP_CHECK(hipDeviceSynchronize());
        }

        // exchange data
        // GPU0 [0, mid) -> GPU1
        // GPU1 [mid, n) -> GPU0
        HIP_CHECK(hipMemcpyPeer(d_qx[1][next], 1, d_qx[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qy[1][next], 1, d_qy[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qz[1][next], 1, d_qz[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qx[0][next] + mid, 0, d_qx[1][next] + mid, 1, (n - mid) * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qy[0][next] + mid, 0, d_qy[1][next] + mid, 1, (n - mid) * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qz[0][next] + mid, 0, d_qz[1][next] + mid, 1, (n - mid) * sizeof(double)));

        // chech if hit
        double p_qx, p_qy, p_qz;
        double a_qx, a_qy, a_qz;
        HIP_CHECK(hipMemcpy(&p_qx, d_qx[0][next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&p_qy, d_qy[0][next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&p_qz, d_qz[0][next] + planet, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&a_qx, d_qx[0][next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&a_qy, d_qy[0][next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&a_qz, d_qz[0][next] + asteroid, sizeof(double), hipMemcpyDeviceToHost));
        double dx = p_qx - a_qx;
        double dy = p_qy - a_qy;
        double dz = p_qz - a_qz;
        // double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius)
        {
            hit_time_step = step;
            break;
        }

        // Swap buffers for double buffering
        std::swap(cur, next);
    }

    // clean up
    for (int gpu = 0; gpu < 2; gpu++)
    {
        HIP_CHECK(hipSetDevice(gpu));
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipFree(d_qx[gpu][buf]));
            HIP_CHECK(hipFree(d_qy[gpu][buf]));
            HIP_CHECK(hipFree(d_qz[gpu][buf]));
        }
        HIP_CHECK(hipFree(d_vx[gpu]));
        HIP_CHECK(hipFree(d_vy[gpu]));
        HIP_CHECK(hipFree(d_vz[gpu]));
        HIP_CHECK(hipFree(d_m[gpu]));
        HIP_CHECK(hipFree(d_type[gpu]));
    }
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
    int &gravity_device_id, double &missile_cost)
{
    // Split workload
    int mid = n / 2;
    int ranges[2][2] = {{0, mid}, {mid, n}};
    int counts[2] = {mid, n - mid};

    // problem 3 specific
    std::vector<int> device_indices;
    std::vector<double> tmp_qx, tmp_qy, tmp_qz, tmp_vx, tmp_vy, tmp_vz, tmp_m;
    for (int i = 0; i < n; i++)
    {
        if (h_type[i] == DEVICE_TYPE)
        {
            device_indices.push_back(i);
        }
    }

    // Device pointers
    double *d_qx[2][2], *d_qy[2][2], *d_qz[2][2]; // [gpu][buffer_idx]
    double *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_m[2];
    int *d_type[2];

    // Allocate device memory and copy data H2D
    for (int gpu = 0; gpu < 2; gpu++)
    {
        HIP_CHECK(hipSetDevice(gpu));

        // allocate positions with double buffering
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipMalloc(&d_qx[gpu][buf], n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qy[gpu][buf], n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qz[gpu][buf], n * sizeof(double)));
        }
        // allocate velocities
        HIP_CHECK(hipMalloc(&d_vx[gpu], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy[gpu], n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz[gpu], n * sizeof(double)));
        // allocate masses
        HIP_CHECK(hipMalloc(&d_m[gpu], n * sizeof(double)));
        // allocate types
        HIP_CHECK(hipMalloc(&d_type[gpu], n * sizeof(int)));

        // H2D copies
        HIP_CHECK(hipMemcpy(d_qx[gpu][0], h_qx.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[gpu][0], h_qy.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[gpu][0], h_qz.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx[gpu], h_vx.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy[gpu], h_vy.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz[gpu], h_vz.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m[gpu], h_m.data(), n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_type[gpu], h_type.data(), n * sizeof(int), hipMemcpyHostToDevice));
    }

    int cur = 0;
    int next = 1;
    int blockSize = 256;

    for (int step = 1; step < param::n_steps; step++)
    {
        // Launch kernels on both GPUs
        for (int gpu = 0; gpu < 2; gpu++)
        {
            HIP_CHECK(hipSetDevice(gpu));
            int count = counts[gpu];
            int gridSize = (count + blockSize - 1) / blockSize;

            run_step_kernel<<<gridSize, blockSize>>>(
                step, n,
                ranges[gpu][0], ranges[gpu][1],
                param::dt, param::G, param::eps,
                d_qx[gpu][cur], d_qy[gpu][cur], d_qz[gpu][cur],
                d_qx[gpu][next], d_qy[gpu][next], d_qz[gpu][next],
                d_vx[gpu], d_vy[gpu], d_vz[gpu],
                d_m[gpu], d_type[gpu]);
            HIP_CHECK(hipGetLastError());
        }

        // synchronize
        for (int gpu = 0; gpu < 2; gpu++)
        {
            HIP_CHECK(hipSetDevice(gpu));
            HIP_CHECK(hipDeviceSynchronize());
        }

        // exchange data
        // GPU0 [0, mid) -> GPU1
        // GPU1 [mid, n) -> GPU0
        HIP_CHECK(hipMemcpyPeer(d_qx[1][next], 1, d_qx[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qy[1][next], 1, d_qy[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qz[1][next], 1, d_qz[0][next], 0, mid * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qx[0][next] + mid, 0, d_qx[1][next] + mid, 1, (n - mid) * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qy[0][next] + mid, 0, d_qy[1][next] + mid, 1, (n - mid) * sizeof(double)));
        HIP_CHECK(hipMemcpyPeer(d_qz[0][next] + mid, 0, d_qz[1][next] + mid, 1, (n - mid) * sizeof(double)));

        // check if hit
        std::vector<double> h_qx(n), h_qy(n), h_qz(n);
        std::vector<double> h_vx(n), h_vy(n), h_vz(n);
        std::vector<double> h_m(n);
        HIP_CHECK(hipMemcpy(h_qx.data(), d_qx[0][next], n * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_qy.data(), d_qy[0][next], n * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_qz.data(), d_qz[0][next], n * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_vx.data(), d_vx[0], mid * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_vx.data() + mid, d_vx[1] + mid, (n - mid) * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_vy.data(), d_vy[0], mid * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_vy.data() + mid, d_vy[1] + mid, (n - mid) * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_vz.data(), d_vz[0], mid * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_vz.data() + mid, d_vz[1] + mid, (n - mid) * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_m.data(), d_m[0], n * sizeof(double), hipMemcpyDeviceToHost));

        double dx = h_qx[planet] - h_qx[asteroid];
        double dy = h_qy[planet] - h_qy[asteroid];
        double dz = h_qz[planet] - h_qz[asteroid];
        // if the planet is hit
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius)
        {
            // if no device can prevent the hit
            if (missile_cost == -999)
            {
                gravity_device_id = -1;
                missile_cost = 0.0;
            }
            break;
        }

        auto distance = [&](int i, int j) -> double
        {
            double dx = h_qx[i] - h_qx[j];
            double dy = h_qy[i] - h_qy[j];
            double dz = h_qz[i] - h_qz[j];
            return sqrt(dx * dx + dy * dy + dz * dz);
        };

        double missile_dist = step * param::dt * param::missile_speed;
        for (auto it = device_indices.begin(); it != device_indices.end();)
        {
            int device_id = *it;
            double device_dist = distance(planet, device_id);
            // missle hit the device
            if (missile_dist > device_dist)
            {
                double cost = (step + 1) * param::dt * 1000.0 + 100000.0;
                // copy current simulation state to temp
                copy_state(
                    tmp_qx, tmp_qy, tmp_qz,
                    tmp_vx, tmp_vy, tmp_vz,
                    tmp_m,
                    h_qx, h_qy, h_qz,
                    h_vx, h_vy, h_vz,
                    h_m);

                // set hit device mass to 0
                tmp_m[device_id] = 0;

                // continue simulation to see if asteroid hits planet
                // this is basically problem 2 but starts with step + 1
                // for (int s = step + 1; s <= param::n_steps; s++)
                // {
                //     run_step(s, n, tmp_qx, tmp_qy, tmp_qz, tmp_vx, tmp_vy, tmp_vz, tmp_m, type);
                //     double ddx = tmp_qx[planet] - tmp_qx[asteroid];
                //     double ddy = tmp_qy[planet] - tmp_qy[asteroid];
                //     double ddz = tmp_qz[planet] - tmp_qz[asteroid];
                //     if (ddx * ddx + ddy * ddy + ddz * ddz < param::planet_radius * param::planet_radius)
                //     {
                //         // asteroid still hits planet
                //         cost = -999;
                //         break;
                //     }
                // }
                int hit_time_step = -2;
                solve_p2(n, planet, asteroid,
                         tmp_qx, tmp_qy, tmp_qz,
                         tmp_vx, tmp_vy, tmp_vz,
                         tmp_m, h_type, hit_time_step,
                         step + 1);
                // asteroid still hits planet
                if (hit_time_step != -2)
                {
                    cost = -999;
                }

                // update minimum cost and device id
                if (cost != -999 && (missile_cost == -999 || cost < missile_cost))
                {
                    missile_cost = cost;
                    gravity_device_id = device_id;
                }

                // remove device id from consideration
                it = device_indices.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // Swap buffers for double buffering
        std::swap(cur, next);
    }

    // clean up
    for (int gpu = 0; gpu < 2; gpu++)
    {
        HIP_CHECK(hipSetDevice(gpu));
        for (int buf = 0; buf < 2; buf++)
        {
            HIP_CHECK(hipFree(d_qx[gpu][buf]));
            HIP_CHECK(hipFree(d_qy[gpu][buf]));
            HIP_CHECK(hipFree(d_qz[gpu][buf]));
        }
        HIP_CHECK(hipFree(d_vx[gpu]));
        HIP_CHECK(hipFree(d_vy[gpu]));
        HIP_CHECK(hipFree(d_vz[gpu]));
        HIP_CHECK(hipFree(d_m[gpu]));
        HIP_CHECK(hipFree(d_type[gpu]));
    }
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
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    std::vector<int> type_int;

    auto distance = [&](int i, int j) -> double
    {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Enable P2P access
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));

    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    convert_type(type, type_int);
    for (int i = 0; i < n; i++)
    {
        if (type_int[i] == DEVICE_TYPE)
        {
            m[i] = 0;
        }
    }
    solve_p1(n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type_int, min_dist);

    // Problem 2
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    convert_type(type, type_int);
    solve_p2(n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type_int, hit_time_step);

    // if there is no need to destroy any gravity device, problem 3 can be skipped
    if (hit_time_step == -2)
    {
        write_output(argv[2], min_dist, hit_time_step, -1, 0);
        return 0;
    }

    // Problem 3
    // TODO
    int gravity_device_id = -999;
    double missile_cost = -999;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    convert_type(type, type_int);
    solve_p3(n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type_int, gravity_device_id, missile_cost);
    // std::vector<int> device_indices;
    // std::vector<double> tmp_qx, tmp_qy, tmp_qz, tmp_vx, tmp_vy, tmp_vz, tmp_m;
    // for (int i = 0; i < n; i++)
    // {
    //     if (type_int[i] == DEVICE_TYPE)
    //     {
    //         device_indices.push_back(i);
    //     }
    // }
    // for (int step = 0; step <= param::n_steps; step++)
    // {
    //     if (step > 0)
    //     {
    //         run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
    //     }
    //     double dx = qx[planet] - qx[asteroid];
    //     double dy = qy[planet] - qy[asteroid];
    //     double dz = qz[planet] - qz[asteroid];

    //     // the planet is hit
    //     if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius)
    //     {
    //         // if no device can prevent the hit
    //         if (missile_cost == -999)
    //         {
    //             gravity_device_id = -1;
    //             missile_cost = 0.0;
    //         }
    //         break;
    //     }

    //     double missile_dist = step * param::dt * param::missile_speed;
    //     for (auto it = device_indices.begin(); it != device_indices.end();)
    //     {
    //         int device_id = *it;
    //         double device_dist = distance(planet, device_id);
    //         // missle hit the device
    //         if (missile_dist > device_dist)
    //         {
    //             double cost = (step + 1) * param::dt * 1000.0 + 100000.0;
    //             // copy current simulation state to temp
    //             copy_state(
    //                 tmp_qx, tmp_qy, tmp_qz,
    //                 tmp_vx, tmp_vy, tmp_vz,
    //                 tmp_m,
    //                 qx, qy, qz,
    //                 vx, vy, vz,
    //                 m);

    //             // set hit device mass to 0
    //             tmp_m[device_id] = 0;

    //             // continue simulation to see if asteroid hits planet
    //             for (int s = step + 1; s <= param::n_steps; s++)
    //             {
    //                 run_step(s, n, tmp_qx, tmp_qy, tmp_qz, tmp_vx, tmp_vy, tmp_vz, tmp_m, type);
    //                 double ddx = tmp_qx[planet] - tmp_qx[asteroid];
    //                 double ddy = tmp_qy[planet] - tmp_qy[asteroid];
    //                 double ddz = tmp_qz[planet] - tmp_qz[asteroid];
    //                 if (ddx * ddx + ddy * ddy + ddz * ddz < param::planet_radius * param::planet_radius)
    //                 {
    //                     // asteroid still hits planet
    //                     cost = -999;
    //                     break;
    //                 }
    //             }

    //             // update minimum cost and device id
    //             if (cost != -999 && (missile_cost == -999 || cost < missile_cost))
    //             {
    //                 missile_cost = cost;
    //                 gravity_device_id = device_id;
    //             }

    //             // remove device id from consideration
    //             it = device_indices.erase(it);
    //         }
    //         else
    //         {
    //             ++it;
    //         }
    //     }
    // }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
