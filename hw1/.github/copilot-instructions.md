Hint
For BFS algorithm

Intel TBB provide thread-safe concurrent access containers. You may find them useful for parallelizing your code. They are installed on Taiwania3. To use them, type module load intel in your terminal and include the correspond header files and compile with -ltbb.
For example, #include "tbb/xxx.h".

boost::functional::hash can be used to hash more STL containers. It’s especially handy when you want to use complex keys in std::unordered_map/std::unordered_set
For example, #include <boost/functional/hash.hpp>
For more detail, see https://www.boost.org/doc/libs/1_55_0b1/doc/html/hash.html

(easy) Try to avoid/detect dead states, e.g.:
######       ######
# .###       # .###
#  ###       #  ###
#X   #       #X   #
#xo  #       #  ox#
#  ###       #  ###
######       ######
(medium-hard) Reduce the number of states by combining person move only states.
(hard) Use an efficient state representation to reduce memory usage
Problem Description
"Sokoban (倉庫番, sōko-ban, "warehouse keeper") is a puzzle video game in which the player pushes crates or boxes around in a warehouse, trying to get them to storage locations." – Wikipedia

You are asked to implement a solver for Sokoban and parallelize it with threads
(either pthread or OpenMP; you can use std::thread as a pthread wrapper).

Input Format
Input is given in an ASCII text file. The filename is given from the command line
in argv[1].

Every line ends with a newline character \n.
Each line represents a row of tiles; every character in the line represents a tile
in the game.

The tiles are listed as follows:

o: The player stepping on a regular tile.
O: The player stepping on a target tile.
x: A box on a regular tile.
X: A box on a target tile.
  (space): Nothing on a regular tile.
.: Nothing on a target tile.
#: Wall.
@: A fragile tile where only the player can step on. The boxes are not allowed to be put on it.
!: The player stepping on a fragile tile.
Your program only need to deal with valid inputs. It is guranteed that:

There is only one player, stepping on either a regular tile, a target tile or a fragile tile.
The number of target tiles are equal to the number of boxes.
All tiles other than wall tiles are within an enclosed area formed by wall tiles.
There is at least one solution.
The size of the map is less than 256 pixels
Output Format
You program should output a sequence of actions that pushes all the boxes to
the target tiles, to stdout. (Do not output anything else to stdout, otherwise
your output may be considered invalid. For debugging purposes, please output to
stderr. Please also remove the debug output from your
submission as they may harm performance)

The output sequence should end with a newline character \n, and contain
only uppercase WASD characters:

W: move the player up
A: move the player left
S: move the player down
D: move the player right
Your solution is not required to have the least number of moves.
Any sequence that solves the problem is valid.

Input Example
This is a valid input:

#########
#  xox..#
#   #####
#########
This input is invalid because there are tiles outside of the wall-enclosed area:

#########
#  xox..#
#   #####
#####   #
This input is invalid because there are fewer target tiles than the boxes:

#########
#  xox .#
#   #####
#########
This input is invalid because there exists no solution:

#########
# ox x..#
#   #####
#########
Output Example
Consider the following problem:

#########
#  xox..#
#   #####
#########
A valid output is:

DDAAASAAWDDDD
Another valid output is:

DDAAADASAAWDDDD
Although the second solution takes more steps then the first one, both output are considered correct and will be accepted.

Execution
Your code will be compiled with the following command:

g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1
We will use make to build your code. The default Makefile for this homework is provided at /work/b11902044/pp25/hw1/Makefile.
If you wish to change the compilation flags, please include Makefile in your submission.

To build your code by make, make sure that both the Makefile and your hw1.cpp are in the working directory. Running the make command line will build hw1
for you. To remove the built files, please execute the make clean command.

Your code will be executed with a command equalviant to:

srun -A ACD114118 -n1 -c${threads} ./hw1 ${input}
where:

${threads} is the number of threads
${input} is the path to the input file
The time limit for each input test case is 30 seconds. ${threads}=6 for all test cases.


Sample Testcases
The sample test cases are located at /work/b11902044/pp25/hw1/samples.
