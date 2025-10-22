pos="4.152 2.398 -2.601"
tarpos="0 0 0"
width=512
height=512


read x1 y1 z1 <<< "$pos"
read x2 y2 z2 <<< "$tarpos"
filename="output_cpu.png"

./hw3_cpu \
    $x1 $y1 $z1 \
    $x2 $y2 $z2 \
    $width $height \
    $filename