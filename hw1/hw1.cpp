#include <iostream>
#include <omp.h>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <string>
#include <algorithm>

using namespace std;

unordered_map<string, pair<int, int>> DYDX = {
    {"W", {-1, 0}},
    {"A", {0, -1}},
    {"S", {1, 0}},
    {"D", {0, 1}},
};

/*
Vertex
2-D Map state compressed into 1-D representation
' ' (space) for empty cell
'#' for wall
'x' for box
'.' for target (nothing on it)
'o' for player
'X' for box on target
'O' for player on target
'@' for fragile tile (no box allowed)
'!' for player on fragile tile
*/
struct Vertex
{
    vector<char> m;
    int width;
    int height;
    int player_loc;
    vector<int> box_locs;    // not updated
    vector<int> target_locs; // not updated
};

char get_tile(const Vertex *vertex, int loc)
{
    if (loc < 0 || loc >= vertex->width * vertex->height)
    {
        throw out_of_range("Location out of bounds");
    }
    return vertex->m[loc];
}

int get_loc_from_xy(const Vertex *vertex, int x, int y)
{
    if (x < 0 || x >= vertex->width || y < 0 || y >= vertex->height)
    {
        throw out_of_range("Coordinates out of bounds");
    }
    return y * vertex->width + x;
}

int get_x_from_loc(const Vertex *vertex, int loc)
{
    if (loc < 0 || loc >= vertex->width * vertex->height)
    {
        throw out_of_range("Location out of bounds");
    }
    return loc % vertex->width;
}
int get_y_from_loc(const Vertex *vertex, int loc)
{
    if (loc < 0 || loc >= vertex->width * vertex->height)
    {
        throw out_of_range("Location out of bounds");
    }
    return loc / vertex->width;
}

void print_position(const Vertex *vertex, int loc)
{
    int x = get_x_from_loc(vertex, loc);
    int y = get_y_from_loc(vertex, loc);
    cerr << "PositionYX: (" << y << ", " << x << ")" << endl;
}

void load_state_from_path(Vertex *vertex, const char *path)
{
    // cerr << "Loading state from testcase path: " << path << endl;

    ifstream input(path);
    if (!input.is_open())
    {
        throw runtime_error("Could not open file");
    }

    int y = 0, width = 0;
    string line;
    while (getline(input, line))
    {
        // cerr << line << endl;
        width = max(width, (int)line.size());

        for (int x = 0; x < line.size(); x++)
        {
            char c = line[x];
            vertex->m.push_back(c);
            if (c == 'o' || c == 'O' || c == '!')
            {
                vertex->player_loc = y * width + x;
            }
            else if (c == 'x' || c == 'X')
            {
                vertex->box_locs.push_back(y * width + x);
            }
            else if (c == '.' || c == 'X' || c == 'O')
            {
                vertex->target_locs.push_back(y * width + x);
            }
        }

        y++;
    }
    vertex->width = width;
    vertex->height = y;
    return;
}

// Graph (Directed)
struct Graph
{
    // Number of edges in the graph
    int num_edges;
    // Number of vertices in the graph
    int num_nodes;

    // The node reached by vertex i's first outgoing edge is given by
    // outgoing_edges[outgoing_starts[i]].  To iterate over all
    // outgoing edges, please see the top-down bfs implementation.
    int *outgoing_starts;
    Vertex *outgoing_edges;

    int *incoming_starts;
    Vertex *incoming_edges;
};

/*
    try to move at the given direction
    return the next game state if the move is valid
    return nullptr otherwise
*/
Vertex *try_push(const Vertex *cur_state, int dy, int dx)
{
    int y = get_y_from_loc(cur_state, cur_state->player_loc);
    int x = get_x_from_loc(cur_state, cur_state->player_loc);

    int yy = y + dy;
    int xx = x + dx;
    int yyy = yy + dy;
    int xxx = xx + dx;

    // Check bounds for the next position
    if (yy < 0 || yy >= cur_state->height || xx < 0 || xx >= cur_state->width)
    {
        return nullptr;
    }

    int next_loc = get_loc_from_xy(cur_state, xx, yy);
    char next_tile = get_tile(cur_state, next_loc);

    Vertex *next_state = new Vertex(*cur_state); // Create a copy of the current state

    // Case 1: Move to an empty space, target, or fragile tile
    if (next_tile == ' ' || next_tile == '.' || next_tile == '@')
    {
        if (next_tile == ' ')
            next_state->m[next_loc] = 'o';
        else if (next_tile == '.')
            next_state->m[next_loc] = 'O';
        else // next_tile == '@'
            next_state->m[next_loc] = '!';
    }
    // Case 2: Push a box
    else if (next_tile == 'x' || next_tile == 'X')
    {
        // Check bounds for the position behind the box
        if (yyy < 0 || yyy >= cur_state->height || xxx < 0 || xxx >= cur_state->width)
        {
            delete next_state;
            return nullptr;
        }
        int after_box_loc = get_loc_from_xy(cur_state, xxx, yyy);
        char after_box_tile = get_tile(cur_state, after_box_loc);

        if (after_box_tile == ' ' || after_box_tile == '.')
        {
            // Update tile where the box was
            if (next_tile == 'x')
                next_state->m[next_loc] = 'o';
            else // next_tile == 'X'
                next_state->m[next_loc] = 'O';

            // Update tile where the box is pushed to
            if (after_box_tile == ' ')
                next_state->m[after_box_loc] = 'x';
            else // after_box_tile == '.'
                next_state->m[after_box_loc] = 'X';
        }
        else
        {
            // Cannot push box (e.g., into a wall or another box or fragile tile)
            delete next_state;
            return nullptr;
        }
    }
    else
    {
        // Invalid move (e.g., into a wall)
        delete next_state;
        return nullptr;
    }

    // Update player's original position
    char original_player_tile = get_tile(cur_state, cur_state->player_loc);
    if (original_player_tile == 'o')
        next_state->m[cur_state->player_loc] = ' ';
    else if (original_player_tile == '!')
        next_state->m[cur_state->player_loc] = '@';
    else // original_player_tile == 'O'
        next_state->m[cur_state->player_loc] = '.';

    // Update player location in the new state
    next_state->player_loc = next_loc;

    // Note: box_locs and target_locs are not updated here as they are not
    // strictly needed for the logic if we rely on the map 'm'.
    // If they are needed for other parts of the solver, they would need to be updated.

    return next_state;
}

void test_try_push(const Vertex *v)
{
    cerr << "\n--- Testing try_push ---" << endl;
    for (auto const &[dir, move] : DYDX)
    {
        cerr << "Trying to move " << dir << "..." << endl;
        Vertex *next_state = try_push(v, move.first, move.second);
        if (next_state)
        {
            cerr << "Move successful. New state:" << endl;
            for (int i = 0; i < next_state->height; ++i)
            {
                for (int j = 0; j < next_state->width; ++j)
                {
                    cerr << next_state->m[i * next_state->width + j];
                }
                cerr << endl;
            }
            delete next_state; // Clean up the allocated memory
        }
        else
        {
            cerr << "Move invalid." << endl;
        }
        cerr << "------------------------" << endl;
    }
}

bool is_solved(const Vertex *v)
{
    // The map is stored as a flattened 1-D array of characters
    for (char c : v->m)
    {
        if (c == 'x')
        { // unsolved box
            return false;
        }
    }
    return true;
}

/* BFS-related */
// Function to convert the map vector to a string for hashing
string state_to_string(const Vertex *v)
{
    return string(v->m.begin(), v->m.end());
}

struct StateInfo
{
    Vertex *parent;
    char move;

    StateInfo() : parent(nullptr), move(0) {}
    StateInfo(Vertex *p, char m) : parent(p), move(m) {}
};

/*
    Solves the puzzle using parallel BFS and reconstructs the path.
    Uses level-synchronous parallelization where all states at each BFS level
    are processed in parallel.
*/
void solve_bfs_with_path(Vertex *start_node)
{
    vector<Vertex *> current_level;
    vector<Vertex *> next_level;

    // Visited map now stores parent state pointer and move character
    unordered_map<string, StateInfo> visited;
    // Keep track of all allocated vertices for cleanup
    vector<Vertex *> allocated_vertices;

    std::string start_str = state_to_string(start_node);
    current_level.push_back(start_node);
    visited[start_str] = StateInfo(nullptr, 0); // Start node has no parent

    bool solved = false;
    Vertex *final_state = nullptr;

    while (!current_level.empty() && !solved)
    {
        next_level.clear();

// Process current level in parallel
#pragma omp parallel
        {
            vector<Vertex *> thread_next_states; // Thread-local storage for new states

#pragma omp for
            for (int i = 0; i < current_level.size(); i++)
            {
                Vertex *current_v = current_level[i];

                // Check if this state is solved
                if (is_solved(current_v))
                {
#pragma omp critical
                    {
                        if (!solved) // Double-check to avoid race condition
                        {
                            final_state = current_v;
                            solved = true;
                        }
                    }
                    continue; // Skip processing moves if solved
                }

                // Try all possible moves
                for (auto const &[dir_str, move] : DYDX)
                {
                    Vertex *next_state = try_push(current_v, move.first, move.second);
                    if (next_state)
                    {
                        string next_state_str = state_to_string(next_state);

                        bool is_new_state = false;
#pragma omp critical
                        {
                            if (visited.find(next_state_str) == visited.end())
                            {
                                // Store the parent pointer and the move
                                visited[next_state_str] = StateInfo(current_v, dir_str[0]);
                                is_new_state = true;
                            }
                        }

                        if (is_new_state)
                        {
                            thread_next_states.push_back(next_state);
                        }
                        else
                        {
                            delete next_state;
                        }
                    }
                }
            }

// Merge thread-local results into global next_level
#pragma omp critical
            {
                next_level.insert(next_level.end(),
                                  thread_next_states.begin(),
                                  thread_next_states.end());
                allocated_vertices.insert(allocated_vertices.end(),
                                          thread_next_states.begin(),
                                          thread_next_states.end());
            }
        }

        // Move to next level
        current_level = std::move(next_level);
    }

    if (solved)
    {
        // Reconstruct path by following parent pointers
        string path = "";
        string current_state_str = state_to_string(final_state);

        while (visited[current_state_str].parent != nullptr)
        {
            path = visited[current_state_str].move + path;
            current_state_str = state_to_string(visited[current_state_str].parent);
        }

        cout << path << endl;
    }
    else
    {
        cerr << "No solution found." << endl;
    }

    // Clean up allocated memory - use delete instead of free
    for (Vertex *v : allocated_vertices)
    {
        delete v;
    }
}

/*
Number of Threads: 6
export OMP_NUM_THREADS=6
*/
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <path_to_testcase>" << endl;
        return 1;
    }
    const char *testcase_path = argv[1];

    Vertex v;
    load_state_from_path(&v, testcase_path);
    solve_bfs_with_path(&v);

    // cerr << "Width: " << v.width << ", Height: " << v.height << endl;
    // cerr << "Player: " << endl;
    // print_position(&v, v.player_loc);
    // cerr << "Boxes: " << endl;
    // for (int loc : v.box_locs)
    // {
    //     print_position(&v, loc);
    // }
    // cerr << "Targets: " << endl;
    // for (int loc : v.target_locs)
    // {
    //     print_position(&v, loc);
    // }

    return 0;
}

// #pragma omp parallel
// {
//     int thread_id = omp_get_thread_num();
//     int num_threads = omp_get_num_threads();
//     cerr << "Hello from thread " << thread_id << " out of " << num_threads << " threads." << endl;
// }