pos="3.726 0.511 -0.096"
tarpos="0 0 0"
width=2048
height=2048


read x1 y1 z1 <<< "$pos"
read x2 y2 z2 <<< "$tarpos"
filename="output_gpu.png"

srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 00:03:00 \
    nvprof \
        ./hw3 \
            $x1 $y1 $z1 \
            $x2 $y2 $z2 \
            $width $height \
            $filename