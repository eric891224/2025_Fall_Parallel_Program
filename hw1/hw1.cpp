#include <iostream>
#include <omp.h>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <stack>
#include <string>
#include <algorithm>

using namespace std;

unordered_map<char, pair<int, int>> DYDX = {
    {'W', {-1, 0}},
    {'A', {0, -1}},
    {'S', {1, 0}},
    {'D', {0, 1}},
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
    // vector<int> box_locs;    // not updated
    // vector<int> target_locs; // not updated
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
            // else if (c == 'x' || c == 'X')
            // {
            //     vertex->box_locs.push_back(y * width + x);
            // }
            // else if (c == '.' || c == 'X' || c == 'O')
            // {
            //     vertex->target_locs.push_back(y * width + x);
            // }
        }

        y++;
    }
    vertex->width = width;
    vertex->height = y;
    return;
}

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

        // check if box pushed into corner deadlock position
        // if (corner_deadlock_locs[get_loc_from_xy(cur_state, xxx, yyy)])
        // {
        //     delete next_state;
        //     return nullptr;
        // }
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

// void test_try_push(const Vertex *v)
// {
//     cerr << "\n--- Testing try_push ---" << endl;
//     for (auto const &[dir, move] : DYDX)
//     {
//         cerr << "Trying to move " << dir << "..." << endl;
//         Vertex *next_state = try_push(v, move.first, move.second);
//         if (next_state)
//         {
//             cerr << "Move successful. New state:" << endl;
//             for (int i = 0; i < next_state->height; ++i)
//             {
//                 for (int j = 0; j < next_state->width; ++j)
//                 {
//                     cerr << next_state->m[i * next_state->width + j];
//                 }
//                 cerr << endl;
//             }
//             delete next_state; // Clean up the allocated memory
//         }
//         else
//         {
//             cerr << "Move invalid." << endl;
//         }
//         cerr << "------------------------" << endl;
//     }
// }

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

// Deadlock detection functions
bool is_corner_deadlock(const Vertex *state, int box_loc)
{
    // If box is on a target, it's not a deadlock
    if (get_tile(state, box_loc) == 'X')
    {
        return false;
    }

    int x = get_x_from_loc(state, box_loc);
    int y = get_y_from_loc(state, box_loc);

    // Only detect the most obvious corner deadlocks - against actual walls (boundaries)
    bool is_top_wall = get_tile(state, get_loc_from_xy(state, x, y - 1)) == '#';
    bool is_bottom_wall = get_tile(state, get_loc_from_xy(state, x, y + 1)) == '#';
    bool is_left_wall = get_tile(state, get_loc_from_xy(state, x - 1, y)) == '#';
    bool is_right_wall = get_tile(state, get_loc_from_xy(state, x + 1, y)) == '#';

    return (is_top_wall && is_left_wall) ||
           (is_top_wall && is_right_wall) ||
           (is_bottom_wall && is_left_wall) ||
           (is_bottom_wall && is_right_wall);
}

bool is_freeze_deadlock(const Vertex *state)
{
    for (int y = 0; y < state->height - 1; y++)
    {
        for (int x = 0; x < state->width - 1; x++)
        {
            // Check 2x2 area
            int tl = get_loc_from_xy(state, x, y);         // top-left
            int tr = get_loc_from_xy(state, x + 1, y);     // top-right
            int bl = get_loc_from_xy(state, x, y + 1);     // bottom-left
            int br = get_loc_from_xy(state, x + 1, y + 1); // bottom-right

            char tl_tile = get_tile(state, tl);
            char tr_tile = get_tile(state, tr);
            char bl_tile = get_tile(state, bl);
            char br_tile = get_tile(state, br);

            // Check if all four positions have boxes
            bool all_boxes = (tl_tile == 'x' || tl_tile == 'X') &&
                             (tr_tile == 'x' || tr_tile == 'X') &&
                             (bl_tile == 'x' || bl_tile == 'X') &&
                             (br_tile == 'x' || br_tile == 'X');

            if (all_boxes)
            {
                // Check if all boxes are on targets (if so, it's not a deadlock)
                bool all_on_targets = (tl_tile == 'X') && (tr_tile == 'X') &&
                                      (bl_tile == 'X') && (br_tile == 'X');

                if (!all_on_targets)
                {
                    return true; // Freeze deadlock detected
                }
            }
        }
    }
    return false;
}

bool is_deadlock_state(const Vertex *state, const vector<bool> &corner_deadlock_locs)
{
    // Be very conservative with deadlock detection to avoid false positives
    // Only check the most obvious deadlocks

    // Check freeze deadlock first (affects multiple boxes)
    if (is_freeze_deadlock(state))
    {
        return true;
    }

    // Check individual box deadlocks - only for boxes not on targets
    for (int loc = 0; loc < state->m.size(); loc++)
    {
        char tile = state->m[loc];
        if (tile == 'x')
        { // Only check boxes not on targets
            // Only use corner deadlock detection, wall deadlock is disabled for now
            if (corner_deadlock_locs[loc])
            {
                return true;
            }
        }
    }

    return false;
}

/* BFS-related */
// Custom hash function for vector<char> to avoid string conversion
struct StateHash
{
    size_t operator()(const vector<char> &state) const
    {
        size_t hash = 0;
        for (size_t i = 0; i < state.size(); ++i)
        {
            hash = hash * 31 + static_cast<size_t>(state[i]);
        }
        return hash;
    }
};

// Function to convert the map vector to a string for hashing (kept for path reconstruction)
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

/* player DFS to pushable boxes */
void dfs_player_reachable(
    Vertex *init_state, 
    vector<Vertex *> &current_level, 
    unordered_map<vector<char>, StateInfo, StateHash> &visited, 
    vector<Vertex *> &allocated_vertices
)
{
    stack<Vertex *> dfs_stack;
    dfs_stack.push(init_state);

    vector<bool> visited_positions(init_state->width * init_state->height, false);
    visited_positions[init_state->player_loc] = true;


    while (!dfs_stack.empty())
    {
        Vertex * current_v = dfs_stack.top();
        dfs_stack.pop();

        int y = get_y_from_loc(current_v, current_v->player_loc);
        int x = get_x_from_loc(current_v, current_v->player_loc);

        for (auto const &[dir_str, move] : DYDX)
        {
            int yy = y + move.first;
            int xx = x + move.second;

            // Check bounds
            if (yy < 0 || yy >= current_v->height || xx < 0 || xx >= current_v->width)
            {
                continue;
            }

            int next_loc = get_loc_from_xy(current_v, xx, yy);
            char next_tile = get_tile(current_v, next_loc);

            // Can move to empty space, target, or fragile tile
            if ((next_tile == ' ' || next_tile == '.' || next_tile == '@') && !visited_positions[next_loc])
            {
                visited_positions[next_loc] = true;
                // Create new state with updated player position
                Vertex *new_state = new Vertex(*current_v);
                dfs_stack.push(new_state);

                // Update player's original position
                char original_player_tile = get_tile(current_v, current_v->player_loc);
                if (original_player_tile == 'o')
                    new_state->m[current_v->player_loc] = ' ';
                else if (original_player_tile == '!')
                    new_state->m[current_v->player_loc] = '@';
                else // original_player_tile == 'O'
                    new_state->m[current_v->player_loc] = '.';

                // Update new player position
                if (next_tile == ' ')
                    new_state->m[next_loc] = 'o';
                else if (next_tile == '.')
                    new_state->m[next_loc] = 'O';
                else // next_tile == '@'
                    new_state->m[next_loc] = '!';

                new_state->player_loc = next_loc;

                /* check if boxes around are pushable */
                bool can_push_box = false;
                for (auto const &[dir_str, move] : DYDX)
                {
                    int box_y = yy + move.first;
                    int box_x = xx + move.second;

                    // Check bounds
                    if (box_y < 0 || box_y >= current_v->height || box_x < 0 || box_x >= current_v->width)
                    {
                        continue;
                    }

                    int box_loc = get_loc_from_xy(current_v, box_x, box_y);
                    char box_tile = get_tile(current_v, box_loc);

                    // If there's a box, try to push it
                    if (box_tile == 'x' || box_tile == 'X')
                    {
                        Vertex *pushed_state = try_push(new_state, move.first, move.second);
                        if (pushed_state)
                        {
                            can_push_box = true;
                            delete pushed_state;
                            break;
                        }
                    }
                }

                // Store in visited and current level
                vector<char> state_vec = new_state->m;
                if (visited.find(state_vec) == visited.end())
                {
                    visited[state_vec] = StateInfo(const_cast<Vertex *>(current_v), dir_str);
                    if (can_push_box) {
                        // print_position(new_state, new_state->player_loc);
                        current_level.push_back(new_state);
                    }
                    allocated_vertices.push_back(new_state);
                }
                else
                {
                    delete new_state; // Already visited
                }
            }
        }
    }
    return;
}

/*
    Solves the puzzle using parallel BFS and reconstructs the path.
    Uses level-synchronous parallelization where all states at each BFS level
    are processed in parallel.
*/
void solve_bfs_with_path(Vertex *start_node)
{
    vector<Vertex *> current_level;
    vector<Vertex *> next_level;

    vector<bool> corner_deadlock_locs(start_node->width * start_node->height, false);
    // Precompute corner deadlock positions
#pragma omp parallel for
    for (int loc = 0; loc < start_node->m.size(); loc++)
    {
        char tile = start_node->m[loc];
        if (tile == ' ' || tile == 'o')
        {
            corner_deadlock_locs[loc] = is_corner_deadlock(start_node, loc);
        }
    }

    // Pre-allocate vectors for better performance
    current_level.reserve(10000);
    next_level.reserve(50000);

    // Use vector<char> directly as key instead of string for better performance
    unordered_map<vector<char>, StateInfo, StateHash> visited;
    // Keep track of all allocated vertices for cleanup
    vector<Vertex *> allocated_vertices;

    current_level.push_back(start_node);
    visited[start_node->m] = StateInfo(nullptr, 0); // Start node has no parent

    dfs_player_reachable(start_node, current_level, visited, allocated_vertices);

    bool solved = false;
    Vertex *final_state = nullptr;

    while (!current_level.empty() && !solved)
    {
        next_level.clear();

        // Use thread-local storage to reduce critical sections
        vector<vector<Vertex *>> thread_local_states(omp_get_max_threads());
        vector<vector<pair<vector<char>, StateInfo>>> thread_local_visited(omp_get_max_threads());

#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic, 1)
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
                        // Add deadlock detection to prune dead branches
                        if (is_deadlock_state(next_state, corner_deadlock_locs))
                        {
                            delete next_state; // Prune this branch
                            continue;
                        }

                        // Store in thread-local containers first
                        thread_local_states[thread_id].push_back(next_state);
                        thread_local_visited[thread_id].emplace_back(
                            next_state->m, StateInfo(current_v, dir_str));
                    }
                }
            }
        }

        // Single critical section to merge all thread results
        for (int t = 0; t < omp_get_max_threads(); t++)
        {
            for (int i = 0; i < thread_local_states[t].size(); i++)
            {
                Vertex *state = thread_local_states[t][i];
                const vector<char> &state_vec = thread_local_visited[t][i].first;

                if (visited.find(state_vec) == visited.end())
                {
                    visited[state_vec] = thread_local_visited[t][i].second;
                    next_level.push_back(state);
                    allocated_vertices.push_back(state);
                }
                else
                {
                    delete state;
                }
            }
        }

        // Move to next level
        current_level = std::move(next_level);
    }

    if (solved)
    {
        // Reconstruct path by following parent pointers
        string path = "";
        vector<char> current_state_vec = final_state->m;

        while (visited[current_state_vec].parent != nullptr)
        {
            path = visited[current_state_vec].move + path;
            current_state_vec = visited[current_state_vec].parent->m;
        }

        cout << path << endl;
    }
    else
    {
        cerr << "No solution found." << endl;
    }

    // Clean up allocated memory - use delete instead of free
#pragma omp parallel for
    for (int i = 0; i < allocated_vertices.size(); i++)
    {
        delete allocated_vertices[i];
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