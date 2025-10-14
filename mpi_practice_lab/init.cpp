#include <mpi.h>
#include <iostream>

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, 
    int dest,int tag, MPI_Comm comm)
int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
    int source, int tag, MPI_Comm comm, MPI_Status *status)

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // Get world size and rank
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // compute
    if (rank < world_size / 2)
        std::cout << "1\n";
    else
        std::cout << "2\n";

    MPI_Finalize();

    return 0;
}
