pos="3.726 0.511 -0.096"
tarpos="0 0 0"
width=2048
height=2048


read x1 y1 z1 <<< "$pos"
read x2 y2 z2 <<< "$tarpos"
filename="output_gpu.png"

./hw3 \
    $x1 $y1 $z1 \
    $x2 $y2 $z2 \
    $width $height \
    $filename