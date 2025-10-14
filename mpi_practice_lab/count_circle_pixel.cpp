#include <mpi.h>
#include <iostream>
#include <assert.h>
// #include <stdio.h>
#include <math.h>
#include <vector>

using namespace std;

// int MPI_Send(const void *buf, int count, MPI_Datatype datatype, 
//     int dest,int tag, MPI_Comm comm)
// int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
//     int source, int tag, MPI_Comm comm, MPI_Status *status)
// int MPI_Reduce(const void *sendbuf, void *recvbuf, 
//     int count, MPI_Datatype datatype, MPI_Op op, 
//     int root, MPI_Comm comm)

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
    MPI_Init(&argc, &argv);
    // Get world size and rank
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	// unsigned long long total_pixels = 0;
    unsigned long long local_pixels = 0;

    // divide work
    unsigned long long chunk_size = r / world_size;
    unsigned long long start = rank * chunk_size;
    unsigned long long end = (rank == world_size - 1) ? r : start + chunk_size;
    unsigned long long y;


    // 計算所有x值從0到r-1的像素點
	for (unsigned long long x = start; x < end; x++) {
		y = ceil(sqrtl(r*r - x*x));
		local_pixels += y;
		// local_pixels %= k;
	}
    local_pixels %= k;

    unsigned long long total_pixels = 0;
    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // printf("%llu\n", (4 * total_pixels) % k);
    if (rank == 0) {
        cout << (4 * total_pixels) % k << endl;
    }
    MPI_Finalize();
    return 0;
}