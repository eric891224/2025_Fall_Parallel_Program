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

void load_state_from_path(Vertex *vertex, const char *path)
{
    ifstream input(path);
    if (!input.is_open())
    {
        throw runtime_error("Could not open file");
    }

    vector<string> lines;
    string line;
    int width = 0;
    
    // First pass: read all lines and calculate width
    while (getline(input, line))
    {
        lines.push_back(line);
        width = max(width, (int)line.size());
    }
    
    vertex->width = width;
    vertex->height = lines.size();
    
    // Second pass: build the map with consistent width
    for (int y = 0; y < lines.size(); y++)
    {
        for (int x = 0; x < width; x++)
        {
            char c = (x < lines[y].size()) ? lines[y][x] : ' ';
            vertex->m.push_back(c);
            if (c == 'o' || c == 'O' || c == '!')
            {
                vertex->player_loc = y * width + x;
            }
        }
    }
    return;
}

/*
    try to pull a box (backward search)
    Simulates the reverse of: player at (prev_y, prev_x) pushed box from (current_y, current_x) to (next_y, next_x)
    In current state: player is at (current_y, current_x), no box here
    In previous state: player was at (prev_y, prev_x), box was at (current_y, current_x)  
*/
Vertex *try_pull(const Vertex *cur_state, int dy, int dx, const vector<bool> &corner_deadlock_locs)
{
    int y = get_y_from_loc(cur_state, cur_state->player_loc);
    int x = get_x_from_loc(cur_state, cur_state->player_loc);

    // Where player came from (reverse direction)
    int prev_y = y - dy;
    int prev_x = x - dx;
    
    // Where box was pushed to (forward direction) 
    int box_dest_y = y + dy;
    int box_dest_x = x + dx;

    // Check bounds
    if (prev_y < 0 || prev_y >= cur_state->height || prev_x < 0 || prev_x >= cur_state->width ||
        box_dest_y < 0 || box_dest_y >= cur_state->height || box_dest_x < 0 || box_dest_x >= cur_state->width)
    {
        return nullptr;
    }

    int prev_loc = get_loc_from_xy(cur_state, prev_x, prev_y);
    int box_dest_loc = get_loc_from_xy(cur_state, box_dest_x, box_dest_y);
    
    char prev_tile = get_tile(cur_state, prev_loc);
    char box_dest_tile = get_tile(cur_state, box_dest_loc);
    char current_tile = get_tile(cur_state, cur_state->player_loc);

    // Previous position must be empty or target
    if (prev_tile != ' ' && prev_tile != '.' && prev_tile != '@')
    {
        return nullptr;
    }

    // Box destination must currently have a box
    if (box_dest_tile != 'x' && box_dest_tile != 'X')
    {
        return nullptr;
    }

    // Current position must be empty or target (where box came from)
    if (current_tile != 'o' && current_tile != 'O' && current_tile != '!')
    {
        return nullptr;
    }

    Vertex *prev_state = new Vertex(*cur_state);

    // Place player at previous position
    if (prev_tile == ' ')
    {
        prev_state->m[prev_loc] = 'o';
    }
    else if (prev_tile == '.')
    {
        prev_state->m[prev_loc] = 'O';
    }
    else // prev_tile == '@'
    {
        prev_state->m[prev_loc] = '!';
    }

    // Place box at current position (where it came from)
    if (current_tile == 'o')
    {
        prev_state->m[cur_state->player_loc] = 'x';
    }
    else if (current_tile == 'O')
    {
        prev_state->m[cur_state->player_loc] = 'X';
    }
    else // current_tile == '!'
    {
        prev_state->m[cur_state->player_loc] = 'x'; // Box on fragile tile
    }

    // Remove box from destination (restore underlying tile)
    if (box_dest_tile == 'x')
    {
        prev_state->m[box_dest_loc] = ' ';
    }
    else // box_dest_tile == 'X'
    {
        prev_state->m[box_dest_loc] = '.';
    }

    prev_state->player_loc = prev_loc;

    return prev_state;
}

/*
    try to move player without pushing a box (backward search)
    return the previous game state if the move is valid
    return nullptr otherwise
*/
Vertex *try_move(const Vertex *cur_state, int dy, int dx)
{
    int y = get_y_from_loc(cur_state, cur_state->player_loc);
    int x = get_x_from_loc(cur_state, cur_state->player_loc);

    // Where the player came from (opposite direction)
    int prev_y = y + dy;
    int prev_x = x + dx;

    // Check bounds
    if (prev_y < 0 || prev_y >= cur_state->height || 
        prev_x < 0 || prev_x >= cur_state->width)
    {
        return nullptr;
    }

    int prev_loc = get_loc_from_xy(cur_state, prev_x, prev_y);
    char prev_tile = get_tile(cur_state, prev_loc);
    char current_tile = get_tile(cur_state, cur_state->player_loc);

    // Previous position must be empty, target, or fragile tile
    if (prev_tile != ' ' && prev_tile != '.' && prev_tile != '@')
    {
        return nullptr;
    }

    Vertex *prev_state = new Vertex(*cur_state);

    // Move player to previous position
    if (prev_tile == ' ')
    {
        prev_state->m[prev_loc] = 'o';
    }
    else if (prev_tile == '.')
    {
        prev_state->m[prev_loc] = 'O';
    }
    else // prev_tile == '@'
    {
        prev_state->m[prev_loc] = '!';
    }

    // Remove player from current position (restore underlying tile)
    if (current_tile == 'o')
    {
        prev_state->m[cur_state->player_loc] = ' ';
    }
    else if (current_tile == 'O')
    {
        prev_state->m[cur_state->player_loc] = '.';
    }
    else // current_tile == '!'
    {
        prev_state->m[cur_state->player_loc] = '@';
    }

    prev_state->player_loc = prev_loc;

    return prev_state;
}

bool is_solved(const Vertex *v)
{
    for (char c : v->m)
    {
        if (c == 'x')
        {
            return false;
        }
    }
    return true;
}

// Deadlock detection functions
bool is_corner_deadlock(const Vertex *state, int box_loc)
{
    if (get_tile(state, box_loc) == 'X')
    {
        return false;
    }

    int x = get_x_from_loc(state, box_loc);
    int y = get_y_from_loc(state, box_loc);

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
            int tl = get_loc_from_xy(state, x, y);
            int tr = get_loc_from_xy(state, x + 1, y);
            int bl = get_loc_from_xy(state, x, y + 1);
            int br = get_loc_from_xy(state, x + 1, y + 1);

            char tl_tile = get_tile(state, tl);
            char tr_tile = get_tile(state, tr);
            char bl_tile = get_tile(state, bl);
            char br_tile = get_tile(state, br);

            bool all_boxes = (tl_tile == 'x' || tl_tile == 'X') &&
                             (tr_tile == 'x' || tr_tile == 'X') &&
                             (bl_tile == 'x' || bl_tile == 'X') &&
                             (br_tile == 'x' || br_tile == 'X');

            if (all_boxes)
            {
                bool all_on_targets = (tl_tile == 'X') && (tr_tile == 'X') &&
                                      (bl_tile == 'X') && (br_tile == 'X');

                if (!all_on_targets)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool is_deadlock_state(const Vertex *state, const vector<bool> &corner_deadlock_locs)
{
    if (is_freeze_deadlock(state))
    {
        return true;
    }

    for (int loc = 0; loc < state->m.size(); loc++)
    {
        char tile = state->m[loc];
        if (tile == 'x')
        {
            if (corner_deadlock_locs[loc])
            {
                return true;
            }
        }
    }

    return false;
}

// Custom hash function for vector<char>
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

struct StateInfo
{
    Vertex *parent;
    char move;

    StateInfo() : parent(nullptr), move(0) {}
    StateInfo(Vertex *p, char m) : parent(p), move(m) {}
};

// Generate all possible solved states for backward search
vector<Vertex*> generate_solved_states(const Vertex* template_state)
{
    vector<Vertex*> solved_states;
    
    // Find all target positions and count boxes
    vector<int> targets;
    int box_count = 0;
    
    for (int i = 0; i < template_state->m.size(); i++)
    {
        char tile = template_state->m[i];
        if (tile == '.' || tile == 'X' || tile == 'O')
        {
            targets.push_back(i);
        }
        if (tile == 'x' || tile == 'X')
        {
            box_count++;
        }
    }
    
    // cerr << "Found " << targets.size() << " targets and " << box_count << " boxes" << endl;
    // cerr << "Target positions: ";
    // for (int target : targets) {
    //     cerr << target << " ";
    // }
    // cerr << endl;
    
    // Only generate solved states if we have enough targets for all boxes
    if (targets.size() < box_count)
    {
        cerr << "Error: Not enough targets (" << targets.size() << ") for boxes (" << box_count << ")" << endl;
        return solved_states;
    }
    
    // Generate solved states with all boxes on targets and player in different positions
    vector<int> valid_player_positions;
    
    for (int i = 0; i < template_state->m.size(); i++)
    {
        char tile = template_state->m[i];
        // Player can only be on non-wall, non-target positions, or on targets if we have extras
        if (tile == ' ' || tile == 'o' || tile == '@' || tile == '!')
        {
            valid_player_positions.push_back(i);
        }
        // Can also be on targets if we have more targets than boxes
        else if ((tile == '.' || tile == 'O') && (int)targets.size() > box_count)
        {
            valid_player_positions.push_back(i);
        }
    }
    
    for (int player_pos : valid_player_positions)
    {
        Vertex* solved = new Vertex(*template_state);
        
        // Start with the template and modify it
        for (int i = 0; i < solved->m.size(); i++)
        {
            char orig_tile = template_state->m[i];
            if (orig_tile == '#' || orig_tile == '@')
            {
                solved->m[i] = orig_tile; // Keep walls and fragile tiles
            }
            else if (orig_tile == '.' || orig_tile == 'X' || orig_tile == 'O')
            {
                solved->m[i] = '.'; // Reset all targets to empty targets first
            }
            else if (orig_tile == 'x' || orig_tile == 'o')
            {
                solved->m[i] = ' '; // Clear boxes and player
            }
            else
            {
                solved->m[i] = ' '; // Clear everything else
            }
        }
        
        // Place exactly 'box_count' boxes on the first 'box_count' targets
        int boxes_placed = 0;
        for (int target_idx = 0; target_idx < targets.size() && boxes_placed < box_count; target_idx++)
        {
            int target_pos = targets[target_idx];
            if (target_pos != player_pos) // Don't place box where player will be
            {
                solved->m[target_pos] = 'X'; // Box on target
                boxes_placed++;
            }
        }
        
        // If we couldn't place all boxes (because player occupied a target), skip
        if (boxes_placed < box_count)
        {
            delete solved;
            continue;
        }
        
        // Place player
        char player_tile = solved->m[player_pos];
        if (player_tile == ' ')
        {
            solved->m[player_pos] = 'o';
        }
        else if (player_tile == '.')
        {
            solved->m[player_pos] = 'O';
        }
        else if (player_tile == '@')
        {
            solved->m[player_pos] = '!';
        }
        
        solved->player_loc = player_pos;
        
        // Verify this is a valid solved state
        if (is_solved(solved))
        {
            // Check if we already have an equivalent state
            bool duplicate = false;
            for (const auto* existing : solved_states)
            {
                if (existing->m == solved->m)
                {
                    duplicate = true;
                    break;
                }
            }
            
            if (!duplicate)
            {
                solved_states.push_back(solved);
            }
            else
            {
                delete solved;
            }
        }
        else
        {
            delete solved;
        }
    }
    
    return solved_states;
}

bool states_equal(const Vertex* a, const Vertex* b)
{
    return a->m == b->m && a->player_loc == b->player_loc;
}

// Reverse move mapping for path reconstruction
char reverse_move(char move)
{
    switch(move)
    {
        case 'W': return 'S';
        case 'S': return 'W';
        case 'A': return 'D';
        case 'D': return 'A';
        default: return move;
    }
}

/*
    Solves the puzzle using backward BFS starting from solved states
*/
void solve_backward_bfs(Vertex *start_node)
{
    // cerr << "Initial state:" << endl;
    // for (int y = 0; y < start_node->height; y++) {
    //     for (int x = 0; x < start_node->width; x++) {
    //         cerr << start_node->m[y * start_node->width + x];
    //     }
    //     cerr << endl;
    // }
    // cerr << "Player at: " << start_node->player_loc << endl;

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

    current_level.reserve(10000);
    next_level.reserve(50000);

    unordered_map<vector<char>, StateInfo, StateHash> visited;
    vector<Vertex *> allocated_vertices;

    // Generate all possible solved states
    vector<Vertex*> solved_states = generate_solved_states(start_node);
    
    // cerr << "Generated " << solved_states.size() << " solved states" << endl;
    
    // if (!solved_states.empty()) {
    //     cerr << "First solved state:" << endl;
    //     auto* first_solved = solved_states[0];
    //     for (int y = 0; y < first_solved->height; y++) {
    //         for (int x = 0; x < first_solved->width; x++) {
    //             cerr << first_solved->m[y * first_solved->width + x];
    //         }
    //         cerr << endl;
    //     }
    //     cerr << "Player at: " << first_solved->player_loc << endl;
    // }
    
    // Initialize with solved states
    for (auto* solved : solved_states)
    {
        current_level.push_back(solved);
        visited[solved->m] = StateInfo(nullptr, 0);
        allocated_vertices.push_back(solved);
    }

    bool found = false;
    Vertex *solution_state = nullptr;
    int iteration = 0;
    const int MAX_ITERATIONS = 25; // Increase search depth

    while (!current_level.empty() && !found && iteration < MAX_ITERATIONS)
    {
        // cerr << "Iteration " << iteration++ << ", exploring " << current_level.size() << " states" << endl;
        next_level.clear();

        vector<vector<Vertex *>> thread_local_states(omp_get_max_threads());
        vector<vector<pair<vector<char>, StateInfo>>> thread_local_visited(omp_get_max_threads());

#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < current_level.size(); i++)
            {
                Vertex *current_v = current_level[i];

                // Check if we reached the start state
                if (current_v->m == start_node->m && current_v->player_loc == start_node->player_loc)
                {
#pragma omp critical
                    {
                        if (!found)
                        {
                            solution_state = current_v;
                            found = true;
                        }
                    }
                    continue;
                }
                
                // Debug: check if we're close to start state
                int diff_count = 0;
                vector<int> diff_positions;
                for (int k = 0; k < current_v->m.size(); k++) {
                    if (current_v->m[k] != start_node->m[k]) {
                        diff_count++;
                        diff_positions.push_back(k);
                    }
                }
                if (diff_count == 2 && diff_positions.size() == 2 && 
                    diff_positions[0] == 15 && diff_positions[1] == 16) {
#pragma omp critical
                    {
                        cerr << "Testing moves from the close state..." << endl;
                        for (auto const &[dir_str, move] : DYDX) {
                            Vertex *test_pull = try_pull(current_v, move.first, move.second, corner_deadlock_locs);
                            Vertex *test_move = try_move(current_v, move.first, move.second);
                            cerr << "Direction " << dir_str << ": pull=" << (test_pull ? "OK" : "NULL") 
                                 << " move=" << (test_move ? "OK" : "NULL") << endl;
                            if (test_pull) delete test_pull;
                            if (test_move) delete test_move;
                        }
                    }
                }

                // Try all reverse moves (pulling boxes and simple moves)
                for (auto const &[dir_str, move] : DYDX)
                {
                    // Try pulling a box
                    Vertex *prev_state = try_pull(current_v, move.first, move.second, corner_deadlock_locs);
                    if (prev_state)
                    {
                        if (!is_deadlock_state(prev_state, corner_deadlock_locs))
                        {
                            thread_local_states[thread_id].push_back(prev_state);
                            thread_local_visited[thread_id].emplace_back(
                                prev_state->m, StateInfo(current_v, dir_str[0]));
                        }
                        else
                        {
                            delete prev_state;
                        }
                    }
                    
                    // Try simple move (no box)
                    prev_state = try_move(current_v, move.first, move.second);
                    if (prev_state)
                    {
                        if (!is_deadlock_state(prev_state, corner_deadlock_locs))
                        {
                            thread_local_states[thread_id].push_back(prev_state);
                            thread_local_visited[thread_id].emplace_back(
                                prev_state->m, StateInfo(current_v, dir_str[0]));
                        }
                        else
                        {
                            delete prev_state;
                        }
                    }
                }
            }
        }

        // Merge thread results
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

        current_level = std::move(next_level);
    }

    if (found)
    {
        // Reconstruct path by reversing both order and direction
        string path = "";
        vector<char> current_state_vec = solution_state->m;

        while (visited[current_state_vec].parent != nullptr)
        {
            char move = visited[current_state_vec].move;
            path = reverse_move(move) + path; // Reverse direction and prepend
            current_state_vec = visited[current_state_vec].parent->m;
        }

        cout << path << endl;
    }
    else
    {
        cerr << "No solution found with backward search." << endl;
    }

    // Cleanup
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
    solve_backward_bfs(&v);

    return 0;
}