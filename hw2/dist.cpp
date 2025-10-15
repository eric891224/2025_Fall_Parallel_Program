#include <mpi.h>

#include "dist.hpp"
#include "sift.hpp"

#include <mpi.h>

void allgather_gradient_pyramid(ScaleSpacePyramid &local_grad_pyramid)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ScaleSpacePyramid complete_pyramid = local_grad_pyramid;

    for (int i = 0; i < local_grad_pyramid.num_octaves; i++)
    {
        for (int j = 0; j < local_grad_pyramid.imgs_per_octave; j++)
        {
            const Image &img = local_grad_pyramid.octaves[i][j];

            // Determine which process computed this image
            int owner = j % size;

            /*
            if rank == owner:
                This process computed this image, broadcast it to others
            else:
                This process didn't compute this image, receive it
            */
            MPI_Bcast(img.data, img.size, MPI_FLOAT, owner, MPI_COMM_WORLD);
            // MPI_Bcast(complete_pyramid.octaves[i][j].data,
            //           img.size, MPI_FLOAT, owner, MPI_COMM_WORLD);
        }
    }

    return;
}