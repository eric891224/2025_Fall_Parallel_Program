#include <cuda.h>
#include <lodepng.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define GLM_FORCE_SWIZZLE // vec3.xyz(), vec3.xyx() ...ect, these are called
                          // "Swizzle".
// https://glm.g-truc.net/0.9.1/api/a00002.html
//
// enable CUDA support
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
//
#include <glm/glm.hpp>
// for the usage of glm functions
// please refer to the document: http://glm.g-truc.net/0.9.9/api/a00143.html
// or you can search on google with typing "glsl xxx"
// xxx is function name (eg. glsl clamp, glsl smoothstep)

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2; // doube precision 2D vector (x, y) or (u, v)
typedef glm::dvec3 vec3; // 3D vector (x, y, z) or (r, g, b)
typedef glm::dvec4 vec4; // 4D vector (x, y, z, w)
typedef glm::dmat3 mat3; // 3x3 matrix

unsigned int num_threads; // number of thread
unsigned int width;       // image width
unsigned int height;      // image height
vec2 iResolution;         // just for convenience of calculation

int AA = 3; // anti-aliasing

double power = 8.0;        // the power of the mandelbulb equation
double md_iter = 24;       // the iteration count of the mandelbulb
double ray_step = 10000;   // maximum step of ray marching
double shadow_step = 1500; // maximum step of shadow casting
double step_limiter = 0.2; // the limit of each step length
double ray_multiplier =
    0.1;                 // prevent over-shooting, lower value for higher quality
double bailout = 2.0;    // escape radius
double eps = 0.0005;     // precision
double FOV = 1.5;        // fov ~66deg
double far_plane = 100.; // scene depth

vec3 camera_pos; // camera position in 3D space (x, y, z)
vec3 target_pos; // target position in 3D space (x, y, z)

unsigned char *raw_image; // 1D image
unsigned char **image;    // 2D image

// save raw_image to PNG file
void write_png(const char *filename)
{
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);

    if (error)
        printf("png error %u: %s\n", error, lodepng_error_text(error));
}

// device constant memory
__constant__ struct
{
    // unsigned int width;
    // unsigned int height;
    vec3 ro; // camera position
    vec3 ta; // target position
    vec3 cf; // forward vector
    vec3 cs; // right (side) vector
    vec3 cu; // up vector
    vec3 sd; // sun direction
    vec3 sc; // light color
    vec2 iResolution;
    int AA;
    double power;
    double md_iter;
    double ray_step;
    double shadow_step;
    double step_limiter;
    double ray_multiplier;
    double bailout;
    double eps;
    double FOV;
    double far_plane;
    // vec3 camera_pos;
    // vec3 target_pos;
} C;

// host mirror of device constant memory
struct ConstData
{
    // unsigned int width;
    // unsigned int height;
    vec3 ro; // camera position
    vec3 ta; // target position
    vec3 cf; // forward vector
    vec3 cs; // right (side) vector
    vec3 cu; // up vector
    vec3 sd; // sun direction
    vec3 sc; // light color
    vec2 iResolution;
    int AA;
    double power;
    double md_iter;
    double ray_step;
    double shadow_step;
    double step_limiter;
    double ray_multiplier;
    double bailout;
    double eps;
    double FOV;
    double far_plane;
    // vec3 camera_pos;
    // vec3 target_pos;
} h_C;

__device__ __forceinline__ double pow_x(double base, int exp)
{
    double result = 1.0;
    int e = exp;
    if (e < 0)
    {
        base = 1.0 / base;
        e = -e;
    }
    while (e > 0)
    {
        if (e & 1)
            result *= base;
        base *= base;
        e >>= 1;
    }
    return result;
}

__device__ __forceinline__ double md(vec3 p, double &trap)
{
    vec3 v = p;
    double dr = 1.;            // |v'|
    double r = glm::length(v); // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    // double r2, r4, r7, r8;

#pragma unroll
    for (int i = 0; i < C.md_iter; ++i)
    {
        // r2 = r * r;
        // r4 = r2 * r2;
        // r7 = r * r2 * r4;
        // r8 = r4 * r4;

        double theta = glm::atan(v.y, v.x) * C.power;
        double phi = glm::asin(v.z / r) * C.power;
        // dr = C.power * glm::pow(r, C.power - 1.) * dr + 1.;
        // dr = C.power * r7 * dr + 1.; // optimized for power = 8
        dr = C.power * pow_x(r, C.power - 1) * dr + 1.;
        // v = p + glm::pow(r, C.power) * vec3(cos(theta) * cos(phi),
        //                                     cos(phi) * sin(theta),
        //                                     -sin(phi)); // update vk+1
        // v = p + r8 * vec3(cos(theta) * cos(phi),
        //                   cos(phi) * sin(theta),
        //                   -sin(phi)); // update vk+1
        v = p + pow_x(r, C.power) * vec3(cos(theta) * cos(phi),
                                         cos(phi) * sin(theta),
                                         -sin(phi)); // update vk+1

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v); // update r
        if (r > C.bailout)
            break; // if escaped
    }
    return 0.5 * log(r) * r / dr; // mandelbulb's DE function
}

// scene mapping
__device__ double map(vec3 p, double &trap, int &ID)
{
    vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));
    vec3 rp = mat3(1., 0., 0., 0., rt.x, -rt.y, 0., rt.y, rt.x) *
              p; // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    ID = 1;
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are
// calculating the surface normal
__device__ double map(vec3 p)
{
    double dmy; // dummy
    int dmy2;   // dummy2
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
__device__ __forceinline__ vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * glm::cos(2. * pi * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
__device__ __forceinline__ double softshadow(vec3 ro, vec3 rd, double k)
{
    double res = 1.0;
    double t = 0.; // total distance
    for (int i = 0; i < C.shadow_step; ++i)
    {
        double h = map(ro + rd * t);
        res = glm::min(res, k * h / t); // closer to the objects, k*h/t terms will
                                        // produce darker shadow
        if (res < 0.02)
            return 0.02;
        t += glm::clamp(h, .001, C.step_limiter); // move ray
    }
    return glm::clamp(res, .02, 1.);
}

// use gradient to calc surface normal
__device__ __forceinline__ vec3 calcNor(vec3 p)
{
    vec2 e = vec2(C.eps, 0.);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()), // dx
                          map(p + e.yxy()) - map(p - e.yxy()), // dy
                          map(p + e.yyx()) - map(p - e.yyx())  // dz
                          ));
}

// first march: find object's surface
__device__ __forceinline__ double trace(vec3 ro, vec3 rd, double &trap,
                                        int &ID)
{
    double t = 0;   // total distance
    double len = 0; // current distance

    for (int i = 0; i < C.ray_step; ++i)
    {
        len = map(ro + rd * t, trap,
                  ID); // get minimum distance from current ray position to the
                       // object's surface
        if (glm::abs(len) < C.eps || t > C.far_plane)
            break;
        t += len * C.ray_multiplier;
    }
    return t < C.far_plane ? t : -1.; // if exceeds the far plane then return -1
                                      // which means the ray missed a shot
}

/* per-pixel render kernel */
__global__ void render(int width, int height, unsigned char *raw_image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    double fcol_r = 0.0;
    double fcol_g = 0.0;
    double fcol_b = 0.0;

    //---anti aliasing
    for (int m = 0; m < C.AA; ++m)
    {
        for (int n = 0; n < C.AA; ++n)
        {
            vec2 p = vec2(x, y) + vec2(m, n) / (double)C.AA;

            //---convert screen space coordinate to (-ap~ap, -1~1)
            // ap = aspect ratio = width/height
            vec2 uv = (-C.iResolution.xy() + 2. * p) / C.iResolution.y;
            uv.y *= -1; // flip upside down
            //---

            //---create camera
            // vec3 ro = camera_pos;               // ray (camera) origin
            // vec3 ta = target_pos;               // target position
            // vec3 cf = glm::normalize(ta - ro);  // forward vector
            // vec3 cs =
            //     glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));  // right (side)
            //     vector
            // vec3 cu = glm::normalize(glm::cross(cs, cf));          // up vector
            vec3 rd = glm::normalize(uv.x * C.cs + uv.y * C.cu +
                                     C.FOV * C.cf); // ray direction
            //---

            //---marching
            double trap; // orbit trap
            int objID;   // the object id intersected with
            double d = trace(C.ro, rd, trap, objID);
            //---

            //---lighting
            vec3 col(0.); // color
            // vec3 sd = glm::normalize(camera_pos);  // sun direction (directional
            // light) vec3 sc = vec3(1., .9, .717);          // light color
            //---

            //---coloring
            if (d < 0.)
            {                   // miss (hit sky)
                col = vec3(0.); // sky color (black)
            }
            else
            {
                vec3 pos = C.ro + rd * d; // hit position
                vec3 nr = calcNor(pos);   // get surface normal
                vec3 hal =
                    glm::normalize(C.sd - rd); // blinn-phong lighting model (vector
                // h)
                // for more info:
                // https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

                // use orbit trap to get the color
                col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.),
                          vec3(.0, .1, .2)); // diffuse color
                vec3 ambc = vec3(0.3);       // ambient color
                double gloss = 32.;          // specular gloss

                // simple blinn phong lighting model
                double amb = (0.7 + 0.3 * nr.y) *
                             (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0,
                                                     1.0));                // self occlution
                double sdw = softshadow(pos + .001 * nr, C.sd, 16.);       // shadow
                double dif = glm::clamp(glm::dot(C.sd, nr), 0., 1.) * sdw; // diffuse
                // double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) *
                //              dif; // self shadow
                double spe = pow_x(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) *
                             dif; // self shadow

                vec3 lin(0.);
                lin += ambc * (.05 + .95 * amb); // ambient color * ambient
                lin += C.sc * dif * 0.8;         // diffuse * light color * light intensity
                col *= lin;

                col = glm::pow(col,
                               vec3(.7, .9, 1.)); // fake SSS (subsurface scattering)
                // col = pow_x(col,
                //                vec3(.7, .9, 1.)); // fake SSS (subsurface scattering)
                col += spe * 0.8; // specular
            }
            //---

            col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.); // gamma correction
            // fcol += vec4(col, 1.);
            fcol_r += col.r;
            fcol_g += col.g;
            fcol_b += col.b;
        }
    }
    //---

    //---color output
    fcol_r = fcol_r / (double)(C.AA * C.AA);
    fcol_g = fcol_g / (double)(C.AA * C.AA);
    fcol_b = fcol_b / (double)(C.AA * C.AA);
    // convert double (0~1) to unsigned char (0~255)
    fcol_r *= 255.0;
    fcol_g *= 255.0;
    fcol_b *= 255.0;
    // write to raw_image
    int idx = (y * width + x) * 4;              // base index for raw_image
    raw_image[idx + 0] = (unsigned char)fcol_r; // r
    raw_image[idx + 1] = (unsigned char)fcol_g; // g
    raw_image[idx + 2] = (unsigned char)fcol_b; // b
    raw_image[idx + 3] = 255;                   // a
                                                //---
}

/* main function */
int main(int argc, char **argv)
{
    // ./hw3 [x1] [y1] [z1] [x2] [y2] [z2] [width] [height] [filename]
    // x1 y1 z1: camera position in 3D space
    // x2 y2 z2: target position in 3D space
    // width height: image size
    // filename: filename
    assert(argc == 10);

    //---init arguments
    camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    width = atoi(argv[7]);
    height = atoi(argv[8]);

    double total_pixel = width * height;
    double current_pixel = 0;

    iResolution = vec2(width, height);
    //---

    //---copy parameters to constant memory
    ConstData hc;
    hc.ro = camera_pos;
    hc.ta = target_pos;
    hc.cf = glm::normalize(hc.ta - hc.ro); // forward vector
    hc.cs = glm::normalize(
        glm::cross(hc.cf, vec3(0., 1., 0.)));         // right (side) vector
    hc.cu = glm::normalize(glm::cross(hc.cs, hc.cf)); // up vector
    hc.sd = glm::normalize(camera_pos);               // sun direction (directional light)
    hc.sc = vec3(1., .9, .717);                       // light color
    hc.iResolution = vec2(width, height);
    hc.AA = AA;
    hc.power = power;
    hc.md_iter = md_iter;
    hc.ray_step = ray_step;
    hc.shadow_step = shadow_step;
    hc.step_limiter = step_limiter;
    hc.ray_multiplier = ray_multiplier;
    hc.bailout = bailout;
    hc.eps = eps;
    hc.FOV = FOV;
    hc.far_plane = far_plane;
    cudaMemcpyToSymbol(C, &hc, sizeof(ConstData));

    // for debugging
    // ConstData check;
    // cudaMemcpyFromSymbol(&check, C, sizeof(ConstData));
    // printf("FOV = %f\n", check.FOV);
    //---

    //---create image
    raw_image = new unsigned char[width * height * 4];
    //---

    //---allocate device memory
    unsigned char *d_raw_image;
    size_t img_size = width * height * 4 * sizeof(unsigned char);
    cudaMalloc(&d_raw_image, img_size);
    //---

    // ---start rendering
    dim3 blockSize(16, 16);
    // dim3 blockSize(8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    render<<<gridSize, blockSize>>>(width, height, d_raw_image);
    cudaDeviceSynchronize();
    //   cudaError_t err = cudaGetLastError();
    //   if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    //   }
    //---

    //---copy image from device to host
    cudaMemcpy(raw_image, d_raw_image, img_size, cudaMemcpyDeviceToHost);
    //---

    //---saving image
    write_png(argv[9]);
    //---

    //---finalize
    delete[] raw_image;
    delete[] image;
    //---

    //---free device memory
    cudaFree(d_raw_image);
    //---

    return 0;
}