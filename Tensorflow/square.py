import numpy as np

def square_data(num_agents, dist_min_thres):
    side_length = np.sqrt(max(1.0, num_agents / 4.0))
    perimeter_length = 4 * side_length

    t = np.linspace(0, perimeter_length, num_agents, endpoint=False)

    def perimeter_position(t_values):
        x = np.zeros_like(t_values)
        y = np.zeros_like(t_values)

        side = side_length

        mask_bottom = (t_values >= 0) & (t_values < side)
        x[mask_bottom] = t_values[mask_bottom]
        y[mask_bottom] = 0

        mask_right = (t_values >= side) & (t_values < 2 * side)
        x[mask_right] = side
        y[mask_right] = t_values[mask_right] - side

        mask_top = (t_values >= 2 * side) & (t_values < 3 * side)
        x[mask_top] = 3 * side - t_values[mask_top]
        y[mask_top] = side

        mask_left = (t_values >= 3 * side) & (t_values < 4 * side)
        x[mask_left] = 0
        y[mask_left] = 4 * side - t_values[mask_left]

        return np.vstack((x, y)).T, mask_bottom, mask_right, mask_top, mask_left

    s_positions, mask_bottom, mask_right, mask_top, mask_left = perimeter_position(t)

    g_positions = np.zeros_like(s_positions)

    g_positions[mask_bottom, 0] = s_positions[mask_bottom, 0]
    g_positions[mask_bottom, 1] = side_length

    g_positions[mask_top, 0] = s_positions[mask_top, 0]
    g_positions[mask_top, 1] = 0

    g_positions[mask_left, 0] = side_length
    g_positions[mask_left, 1] = s_positions[mask_left, 1]

    g_positions[mask_right, 0] = 0
    g_positions[mask_right, 1] = s_positions[mask_right, 1]

    states = np.concatenate([s_positions, np.zeros((num_agents, 2))], axis=1)
    goals = g_positions

    return states, goals