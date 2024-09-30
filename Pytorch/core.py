import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_obstacle_circle(center, radius, num=12):
    theta = np.linspace(0, 2 * np.pi, num=num, endpoint=False).reshape(-1, 1)
    unit_circle = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    circle = np.array(center) + unit_circle * radius
    return circle

def generate_obstacle_rectangle(center, sides, num=12):
    a, b = sides  # side lengths
    n_side_1 = int(num // 2 * a / (a + b))
    n_side_2 = num // 2 - n_side_1
    n_side_3 = n_side_1
    n_side_4 = num - n_side_1 - n_side_2 - n_side_3

    # top
    side_1 = np.concatenate([
        np.linspace(-a / 2, a / 2, n_side_1, endpoint=False).reshape(-1, 1),
        (b / 2) * np.ones(n_side_1).reshape(-1, 1)], axis=1)
    # right
    side_2 = np.concatenate([
        (a / 2) * np.ones(n_side_2).reshape(-1, 1),
        np.linspace(b / 2, -b / 2, n_side_2, endpoint=False).reshape(-1, 1)], axis=1)
    # bottom
    side_3 = np.concatenate([
        np.linspace(a / 2, -a / 2, n_side_3, endpoint=False).reshape(-1, 1),
        (-b / 2) * np.ones(n_side_3).reshape(-1, 1)], axis=1)
    # left
    side_4 = np.concatenate([
        (-a / 2) * np.ones(n_side_4).reshape(-1, 1),
        np.linspace(-b / 2, b / 2, n_side_4, endpoint=False).reshape(-1, 1)], axis=1)

    rectangle = np.concatenate([side_1, side_2, side_3, side_4], axis=0)
    rectangle = rectangle + np.array(center)
    return rectangle

def generate_data(num_agents, dist_min_thres):
    side_length = np.sqrt(max(1.0, num_agents / 8.0))
    states = np.zeros(shape=(num_agents, 2), dtype=np.float32)
    goals = np.zeros(shape=(num_agents, 2), dtype=np.float32)

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(size=(2,)) * side_length
        if i > 0:
            dist_min = np.linalg.norm(states[:i] - candidate, axis=1).min()
            if dist_min <= dist_min_thres:
                continue
        states[i] = candidate
        i += 1

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i]
        if i > 0:
            dist_min = np.linalg.norm(goals[:i] - candidate, axis=1).min()
            if dist_min <= dist_min_thres:
                continue
        goals[i] = candidate
        i += 1

    states = np.concatenate(
        [states, np.zeros(shape=(num_agents, 2), dtype=np.float32)], axis=1)
    return states, goals

class NetworkCBF(nn.Module):
    def __init__(self):
        super(NetworkCBF, self).__init__()
        self.obs_radius = config.OBS_RADIUS

        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x, r, indices=None):
        # Compute norm
        d_norm = torch.sqrt(torch.sum(x[:, :, :2] ** 2 + 1e-4, dim=2, keepdim=True))

        # Identity matrix for self-identification
        eye = torch.eye(x.size(0), device=x.device).unsqueeze(2)
        if eye.dim() < x.dim():
            eye = eye.unsqueeze(0)
        x = torch.cat([x, eye, d_norm - r], dim=2)

        # Remove distant agents
        x, indices = remove_distant_agents(x, indices=indices)

        # Compute distances and masks
        dist = torch.sqrt(torch.sum(x[:, :, :2] ** 2 + 1e-4, dim=2, keepdim=True))
        mask = (dist <= self.obs_radius).float()

        # Pass through convolutional layers
        x = F.relu(self.conv1(x.permute(0, 2, 1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # Output shape: [batch_size, 1, num_agents]
        x = x.permute(0, 2, 1) * mask  # Shape: [batch_size, num_agents, 1]

        return x, mask, indices

class NetworkAction(nn.Module):
    def __init__(self):
        super(NetworkAction, self).__init__()
        self.obs_radius = config.OBS_RADIUS

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        self.fc1 = nn.Linear(128 + 4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, s, g, obs_radius=1.0, indices=None):
        # Compute pairwise differences between all agents
        x = s.unsqueeze(1) - s.unsqueeze(0)  # Shape: [num_agents, num_agents, 4]
        # Identity matrix for self-identification
        eye = torch.eye(x.size(0), device=x.device).unsqueeze(2)
        x = torch.cat([x, eye], dim=2)  # Shape: [num_agents, num_agents, 5]

        # Remove distant agents
        x, indices = remove_distant_agents(x, indices=indices)

        # Compute distances and masks
        dist = torch.norm(x[:, :, :2], dim=2, keepdim=True)
        mask = (dist < obs_radius).float()

        # Pass through convolutional layers
        x = F.relu(self.conv1(x.permute(0, 2, 1)))  # Shape: [batch_size, channels, num_agents]
        x = F.relu(self.conv2(x))

        # Apply masked global max pooling
        x = torch.max(x * mask.permute(0, 2, 1), dim=2)[0]

        # Combine with goal and current velocity information
        x = torch.cat([x, s[:, :2] - g, s[:, 2:]], dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Apply sigmoid activation and rescale
        x = 2.0 * torch.sigmoid(x) + 0.2

        # Split the output and compute gains
        k_1, k_2, k_3, k_4 = torch.split(x, 1, dim=1)
        zeros = torch.zeros_like(k_1)
        gain_x = -torch.cat([k_1, zeros, k_2, zeros], dim=1)
        gain_y = -torch.cat([zeros, k_3, zeros, k_4], dim=1)

        # Compute the action
        state = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
        a_x = torch.sum(state * gain_x, dim=1, keepdim=True)
        a_y = torch.sum(state * gain_y, dim=1, keepdim=True)
        a = torch.cat([a_x, a_y], dim=1)

        return a

def remove_distant_agents(x, indices=None):
    n, _, c = x.size()
    if n <= config.TOP_K:
        return x, None

    if indices is not None:
        # Use the provided indices to select the agents
        x = x[indices[:, 0], indices[:, 1], :]
        x = x.view(n, config.TOP_K, c)
        return x, indices

    d_norm = torch.sqrt(torch.sum(x[:, :, :2] ** 2 + 1e-6, dim=2))
    _, indices = torch.topk(-d_norm, k=config.TOP_K, dim=1)
    row_indices = torch.arange(indices.size(0), device=indices.device).unsqueeze(1).expand(-1, config.TOP_K)
    x = x[row_indices, indices, :]
    indices = torch.stack([row_indices, indices], dim=2).view(-1, 2)
    return x, indices

def dynamics(s, a):
    dsdt = torch.cat([s[:, 2:], a], dim=1)
    return dsdt

def loss_barrier(h, s, r, ttc, model_cbf, indices=None, eps=[1e-3, 0]):
    h_reshape = h.view(-1)
    dang_mask = ttc_dangerous_mask(s, r=r, ttc=ttc, indices=indices)
    dang_mask_reshape = dang_mask.view(-1)
    safe_mask_reshape = ~dang_mask_reshape

    dang_h = h_reshape[dang_mask_reshape]
    safe_h = h_reshape[safe_mask_reshape]

    num_dang = dang_h.size(0)
    num_safe = safe_h.size(0)

    loss_dang = torch.sum(F.relu(dang_h + eps[0])) / (1e-5 + num_dang)
    loss_safe = torch.sum(F.relu(-safe_h + eps[1])) / (1e-5 + num_safe)

    acc_dang = torch.sum((dang_h <= 0).float()) / (1e-5 + num_dang)
    acc_safe = torch.sum((safe_h > 0).float()) / (1e-5 + num_safe)

    acc_dang = acc_dang if num_dang > 0 else torch.tensor(-1.0, device=device)
    acc_safe = acc_safe if num_safe > 0 else torch.tensor(-1.0, device=device)

    return loss_dang, loss_safe, acc_dang, acc_safe

def loss_derivatives(s, a, h, x, r, ttc, alpha, time_step, dist_min_thres, model_cbf, indices=None, eps=[1e-3, 0]):
    dsdt = dynamics(s, a)
    s_next = s + dsdt * time_step

    x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)
    h_next, mask_next, _ = model_cbf(x=x_next, r=dist_min_thres, indices=indices)

    deriv = h_next - h + time_step * alpha * h

    deriv_reshape = deriv.view(-1)
    dang_mask = ttc_dangerous_mask(s=s, r=r, ttc=ttc, indices=indices)
    dang_mask_reshape = dang_mask.view(-1)
    safe_mask_reshape = ~dang_mask_reshape

    dang_deriv = deriv_reshape[dang_mask_reshape]
    safe_deriv = deriv_reshape[safe_mask_reshape]

    num_dang = dang_deriv.size(0)
    num_safe = safe_deriv.size(0)

    loss_dang_deriv = torch.sum(F.relu(-dang_deriv + eps[0])) / (1e-5 + num_dang)
    loss_safe_deriv = torch.sum(F.relu(-safe_deriv + eps[1])) / (1e-5 + num_safe)

    acc_dang_deriv = torch.sum((dang_deriv >= 0).float()) / (1e-5 + num_dang)
    acc_safe_deriv = torch.sum((safe_deriv >= 0).float()) / (1e-5 + num_safe)

    acc_dang_deriv = acc_dang_deriv if num_dang > 0 else torch.tensor(-1.0, device=device)
    acc_safe_deriv = acc_safe_deriv if num_safe > 0 else torch.tensor(-1.0, device=device)

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv

def loss_actions(s, g, a, r, ttc):
    state_gain = -torch.tensor(
        np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32, device=device)

    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = torch.matmul(s_ref, state_gain.t())

    action_ref_norm = torch.sum(action_ref ** 2, dim=1)
    action_net_norm = torch.sum(a ** 2, dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)
    return loss

def statics(s, a, h, alpha, model_cbf):
    dsdt = dynamics(s, a)
    s_next = s + dsdt * config.TIME_STEP

    x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)
    h_next, mask_next, _ = model_cbf(x=x_next, r=config.DIST_MIN_THRES)

    deriv = h_next - h + config.TIME_STEP * alpha * h

    mean_deriv = torch.mean(deriv)
    std_deriv = torch.sqrt(torch.mean((deriv - mean_deriv) ** 2))
    prob_neg = torch.mean((deriv < 0).float())

    return mean_deriv, std_deriv, prob_neg

def ttc_dangerous_mask(s, r, ttc, indices=None):
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # Shape: [N, N, 4]
    eye = torch.eye(s.size(0), device=s.device).unsqueeze(2)
    s_diff = torch.cat([s_diff, eye], dim=2)  # Shape: [N, N, 5]

    s_diff, _ = remove_distant_agents(s_diff, indices=indices)
    x, y, vx, vy, eye = torch.split(s_diff, 1, dim=2)

    x = x + eye
    y = y + eye

    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2

    dist_dangerous = gamma.squeeze(2) < 0

    discriminant = beta ** 2 - 4 * alpha * gamma
    has_two_positive_roots = ((discriminant > 0) & (gamma > 0) & (beta < 0)).squeeze(2)
    root_less_than_ttc = ((-beta - 2 * alpha * ttc) < 0).squeeze(2) | \
                         (((beta + 2 * alpha * ttc) ** 2) < (beta ** 2 - 4 * alpha * gamma)).squeeze(2)
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc

    ttc_dangerous = dist_dangerous | has_root_less_than_ttc
    return ttc_dangerous

def ttc_dangerous_mask_np(s, r, ttc):
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    x, y, vx, vy = np.split(s_diff, 4, axis=2)

    x = x + np.expand_dims(np.eye(s.shape[0]), 2)
    y = y + np.expand_dims(np.eye(s.shape[0]), 2)

    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2

    dist_dangerous = gamma.squeeze(2) < 0

    discriminant = beta ** 2 - 4 * alpha * gamma
    has_two_positive_roots = ((discriminant > 0) & (gamma > 0) & (beta < 0)).squeeze(2)
    root_less_than_ttc = ((-beta - 2 * alpha * ttc) < 0).squeeze(2) | \
                         (((beta + 2 * alpha * ttc) ** 2) < (beta ** 2 - 4 * alpha * gamma)).squeeze(2)
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc

    ttc_dangerous = dist_dangerous | has_root_less_than_ttc
    return ttc_dangerous