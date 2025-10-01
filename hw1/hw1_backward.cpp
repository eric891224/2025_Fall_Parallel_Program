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

void print_position(const Vertex *vertex, int loc)
{
    int x = get_x_from_loc(vertex, loc);
    int y = get_y_from_loc(vertex, loc);
    cerr << "PositionYX: (" << y << ", " << x << ")" << endl;
}

void load_state_from_path(Vertex *vertex, const char *path)
{
    ifstream input(path);
    if (!input.is_open())
    {
        throw runtime_error("Could not open file");
    }

    int y = 0, width = 0;
    string line;
    while (getline(input, line))
    {
        width = max(width, (int)line.size());

        for (int x = 0; x < line.size(); x++)
        {
            char c = line[x];
            vertex->m.push_back(c);
            if (c == 'o' || c == 'O' || c == '!')
            {
                vertex->player_loc = y * width + x;
            }
        }
        y++;
    }
    vertex->width = width;
    vertex->height = y;
    return;
}

/*
    try to move at the given direction (forward search)
    return the next game state if the move is valid
    return nullptr otherwise
*/
Vertex *try_push(const Vertex *cur_state, int dy, int dx, const vector<bool> &corner_deadlock_locs)
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

    Vertex *next_state = new Vertex(*cur_state);

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
        if (corner_deadlock_locs[get_loc_from_xy(cur_state, xxx, yyy)])
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
            delete next_state;
            return nullptr;
        }
    }
    else
    {
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

    next_state->player_loc = next_loc;

    return next_state;
}

/*
    try to pull a box (backward search)
    In backward search, we simulate the reverse of a push operation:
    - Player was at position (y-dy, x-dx)
    - There was a box at position (y+dy, x+dx)
    - Player pushed the box to current player position
    return the previous game state if the pull is valid
    return nullptr otherwise
*/
Vertex *try_pull(const Vertex *cur_state, int dy, int dx, const vector<bool> &corner_deadlock_locs)
{
    int y = get_y_from_loc(cur_state, cur_state->player_loc);
    int x = get_x_from_loc(cur_state, cur_state->player_loc);

    // In reverse: player was at position (y-dy, x-dx)
    int prev_player_y = y - dy;
    int prev_player_x = x - dx;

    // The box was at position (y+dy, x+dx) before being pushed
    int prev_box_y = y + dy;
    int prev_box_x = x + dx;

    // Check bounds
    if (prev_player_y < 0 || prev_player_y >= cur_state->height ||
        prev_player_x < 0 || prev_player_x >= cur_state->width ||
        prev_box_y < 0 || prev_box_y >= cur_state->height ||
        prev_box_x < 0 || prev_box_x >= cur_state->width)
    {
        return nullptr;
    }

    int prev_player_loc = get_loc_from_xy(cur_state, prev_player_x, prev_player_y);
    int prev_box_loc = get_loc_from_xy(cur_state, prev_box_x, prev_box_y);

    char prev_player_tile = get_tile(cur_state, prev_player_loc);
    char prev_box_tile = get_tile(cur_state, prev_box_loc);
    char current_player_tile = get_tile(cur_state, cur_state->player_loc);

    // Previous player position must be empty or target (no box or wall)
    if (prev_player_tile != ' ' && prev_player_tile != '.')
    {
        return nullptr;
    }

    // Previous box position must be empty or target (where box was pulled from)
    if (prev_box_tile != ' ' && prev_box_tile != '.')
    {
        return nullptr;
    }

    // Current player position must have space for a box (player pushed a box here)
    // But since we're going backward, there should be either a player or player on target
    if (current_player_tile != 'o' && current_player_tile != 'O')
    {
        return nullptr;
    }

    Vertex *prev_state = new Vertex(*cur_state);

    // Move player to previous position
    if (prev_player_tile == ' ')
    {
        prev_state->m[prev_player_loc] = 'o';
    }
    else // prev_player_tile == '.'
    {
        prev_state->m[prev_player_loc] = 'O';
    }

    // Place box at its previous position (where it was pulled from)
    if (prev_box_tile == ' ')
    {
        prev_state->m[prev_box_loc] = 'x'; // Box on empty space
    }
    else // prev_box_tile == '.'
    {
        prev_state->m[prev_box_loc] = 'X'; // Box on target
    }

    // Remove box from current player position (restore underlying tile)
    if (current_player_tile == 'o')
    {
        prev_state->m[cur_state->player_loc] = ' '; // Empty space
    }
    else // current_player_tile == 'O'
    {
        prev_state->m[cur_state->player_loc] = '.'; // Target
    }

    prev_state->player_loc = prev_player_loc;

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
vector<Vertex *> generate_solved_states(const Vertex *template_state)
{
    vector<Vertex *> solved_states;

    // Find all target positions and count them
    vector<int> targets;
    for (int i = 0; i < template_state->m.size(); i++)
    {
        if (template_state->m[i] == '.' || template_state->m[i] == 'X' || template_state->m[i] == 'O')
        {
            targets.push_back(i);
        }
    }

    // Generate solved states with different player positions
    // But we need to be more careful - only generate valid reachable positions
    for (int player_pos = 0; player_pos < template_state->m.size(); player_pos++)
    {
        char tile = template_state->m[player_pos];
        // Player can be on empty space, target, or original player position
        if (tile == ' ' || tile == '.' || tile == 'o' || tile == 'O' || tile == 'x')
        {
            Vertex *solved = new Vertex(*template_state);

            // First, create base solved state: all targets have boxes
            for (int i = 0; i < solved->m.size(); i++)
            {
                char orig_tile = template_state->m[i];

                if (orig_tile == '#' || orig_tile == '@') // Keep walls and fragile tiles
                {
                    solved->m[i] = orig_tile;
                }
                else if (orig_tile == '.' || orig_tile == 'X' || orig_tile == 'O')
                {
                    // This is a target position
                    if (i == player_pos)
                    {
                        solved->m[i] = 'O'; // Player on target (no box here for this configuration)
                    }
                    else
                    {
                        solved->m[i] = 'X'; // Box on target
                    }
                }
                else // Empty space, box not on target, or player not on target
                {
                    if (i == player_pos)
                    {
                        solved->m[i] = 'o'; // Player on empty space
                    }
                    else
                    {
                        solved->m[i] = ' '; // Empty space
                    }
                }
            }

            solved->player_loc = player_pos;

            // Only add if this creates a valid solved state
            if (is_solved(solved))
            {
                solved_states.push_back(solved);
            }
            else
            {
                delete solved;
            }
        }
    }

    // If no solved states generated, try a simpler approach
    if (solved_states.empty())
    {
        // Create one basic solved state with all boxes on targets
        Vertex *solved = new Vertex(*template_state);

        // Place all boxes on targets, player on first available space
        int player_placed = false;
        for (int i = 0; i < solved->m.size(); i++)
        {
            char orig_tile = template_state->m[i];

            if (orig_tile == '#' || orig_tile == '@')
            {
                solved->m[i] = orig_tile;
            }
            else if (orig_tile == '.' || orig_tile == 'X' || orig_tile == 'O')
            {
                solved->m[i] = 'X'; // Box on target
            }
            else if (!player_placed && (orig_tile == ' ' || orig_tile == 'o'))
            {
                solved->m[i] = 'o'; // Place player
                solved->player_loc = i;
                player_placed = true;
            }
            else
            {
                solved->m[i] = ' '; // Empty space
            }
        }

        if (player_placed && is_solved(solved))
        {
            solved_states.push_back(solved);
        }
        else
        {
            delete solved;
        }
    }

    return solved_states;
}

bool states_equal(const Vertex *a, const Vertex *b)
{
    return a->m == b->m;
}

// Reverse move mapping for path reconstruction
char reverse_move(char move)
{
    switch (move)
    {
    case 'W':
        return 'S';
    case 'S':
        return 'W';
    case 'A':
        return 'D';
    case 'D':
        return 'A';
    default:
        return move;
    }
}

/*
    Solves the puzzle using backward BFS starting from solved states
*/
void solve_backward_bfs(Vertex *start_node)
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

    current_level.reserve(10000);
    next_level.reserve(50000);

    unordered_map<vector<char>, StateInfo, StateHash> visited;
    vector<Vertex *> allocated_vertices;

    // Generate all possible solved states
    vector<Vertex *> solved_states = generate_solved_states(start_node);

    // Initialize with solved states
    for (auto *solved : solved_states)
    {
        current_level.push_back(solved);
        visited[solved->m] = StateInfo(nullptr, 0);
        allocated_vertices.push_back(solved);
    }

    bool found = false;
    Vertex *solution_state = nullptr;

    while (!current_level.empty() && !found)
    {
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
                if (states_equal(current_v, start_node))
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

                // Try all reverse moves (pulling boxes)
                for (auto const &[dir_str, move] : DYDX)
                {
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
        // Reconstruct path (need to reverse it since we went backward)
        string path = "";
        vector<char> current_state_vec = solution_state->m;

        while (visited[current_state_vec].parent != nullptr)
        {
            char move = visited[current_state_vec].move;
            path = reverse_move(move) + path; // Reverse the move and prepend
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
