#include <mpi.h>
#include <iostream>
#include <assert.h>
// #include <stdio.h>
#include <math.h>
#include <vector>

using namespace std;
/* 
Output format (stdout):
pixels % k

pixels: number of pixels needed to draw the circle
*/
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <radius: int> <k: int>" << endl;
        return 1;
    }
    

    unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long total_pixels = 0;
    unsigned long long y;

    // 計算所有x值從0到r-1的像素點
	for (unsigned long long x = 0; x < r; x++) {
		y = ceil(sqrtl(r*r - x*x));
		total_pixels += y;
		// total_pixels %= k;
	}
    total_pixels %= k;

    // printf("%llu\n", (4 * total_pixels) % k);
    cout << (4 * total_pixels) % k << endl;
    return 0;
}