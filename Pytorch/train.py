import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import core
import config

# For comparison between TensorFlow and PyTorch code
seed = 42

# Setting random seeds for reproducibility
def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU if applicable
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility over speed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args

class AgentDataset(Dataset):
    def __init__(self, num_agents, dist_min_thres, num_samples=1000, device="cpu"):
        super(AgentDataset, self).__init__()
        self.num_agents = num_agents
        self.dist_min_thres = dist_min_thres
        self.num_samples = num_samples
        self.device = device
        self.data = self.generate_data()

    def generate_data(self):
        data = [core.generate_data(self.num_agents, self.dist_min_thres) for _ in range(self.num_samples)]
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        states, goals = self.data[idx]
        states_tensor = torch.tensor(states, dtype=torch.float32)
        goals_tensor = torch.tensor(goals, dtype=torch.float32)
        return states_tensor, goals_tensor

def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        valid_acc = acc[acc[:, i] >= 0, i]
        if len(valid_acc) > 0:
            acc_list.append(np.mean(valid_acc))
        else:
            acc_list.append(0.0)
    return acc_list

def train_epoch(num_agents, model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device):
    model_cbf.train()
    model_action.train()
    total_loss = 0

    accumulation_steps = config.INNER_LOOPS  # Define accumulation steps

    for istep, (states, goals) in enumerate(dataloader):
        if istep >= config.TRAIN_STEPS:
            break  # Limit the training steps

        # Ensure the correct shapes and data movement
        s_np = states.squeeze(0).to(device)
        g_np = goals.squeeze(0).to(device)

        loss_lists_cbf = []
        loss_lists_action = []
        acc_lists_np = []

        steps_accumulated = 0

        # Zero gradients before starting accumulation
        optimizer_cbf.zero_grad()
        optimizer_action.zero_grad()

        # Accumulate losses over multiple steps
        for i in range(accumulation_steps):
            # Compute the control input a_np using the action network
            a_np = model_action(s_np, g_np)

            if np.random.uniform() < config.ADD_NOISE_PROB:
                noise = torch.normal(0, config.NOISE_SCALE, size=a_np.shape).to(device)
                a_np = a_np + noise

            # Simulating the system for one step
            s_np = s_np + torch.cat([s_np[:, 2:], a_np], dim=1) * config.TIME_STEP

            # Compute safety ratio
            s_np_detached = s_np.detach().cpu().numpy()  # Ensure tensor is detached for numpy operations
            safety_mask_np = core.ttc_dangerous_mask_np(
                s_np_detached, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)

            # Convert the safety mask to a tensor
            safety_mask_tensor = torch.tensor(safety_mask_np, dtype=torch.float32).to(device)

            safety_ratio = 1 - torch.mean(safety_mask_tensor, dim=1)
            safety_ratio = torch.mean((safety_ratio == 1).float())

            # CBF and loss calculations
            x = s_np.unsqueeze(1) - s_np.unsqueeze(0)
            h, mask, indices = model_cbf(x, config.DIST_MIN_THRES, indices=None)

            # Compute losses
            loss_action = core.loss_actions(s=s_np, g=g_np, a=a_np, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

            (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv) = core.loss_derivatives(
                s=s_np, a=a_np, h=h, x=x, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION,
                alpha=config.ALPHA_CBF, time_step=config.TIME_STEP, dist_min_thres=config.DIST_MIN_THRES,
                model_cbf=model_cbf)

            (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
                h=h, s=s_np, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION, model_cbf=model_cbf, indices=indices)

            # Accumulate losses
            loss_cbf_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv]
            loss_action_scalar = 0.01 * loss_action

            loss_lists_cbf.append(sum(loss_cbf_list))
            loss_lists_action.append(loss_action_scalar)
            acc_lists_np.append([acc_dang.item(), acc_safe.item(), acc_dang_deriv.item(), acc_safe_deriv.item()])

            steps_accumulated += 1
            if torch.mean(torch.norm(s_np[:, :2] - g_np, dim=1)) < config.DIST_MIN_CHECK:
                break

        # Combining losses for CBF network
        total_loss_cbf = torch.stack(loss_lists_cbf).sum()
        # Manually add weight decay (L2 regularization)
        weight_loss_cbf = [config.WEIGHT_DECAY * (p ** 2).sum() for p in model_cbf.parameters()]
        total_loss_cbf = 10 * (total_loss_cbf + sum(weight_loss_cbf)) / steps_accumulated

        # Backward pass and optimizer step for CBF network
        if (istep // 10) % 2 == 0:
            total_loss_cbf.backward()
            optimizer_cbf.step()
            optimizer_cbf.zero_grad()

        # Combining losses for action network
        total_loss_action = torch.stack(loss_lists_action).sum()
        # Manually add weight decay (L2 regularization)
        weight_loss_action = [config.WEIGHT_DECAY * (p ** 2).sum() for p in model_action.parameters()]
        total_loss_action = 10 * (total_loss_action + sum(weight_loss_action)) / steps_accumulated

        # Backward pass and optimizer step for action network
        if (istep // 10) % 2 != 0:
            total_loss_action.backward()
            optimizer_action.step()
            optimizer_action.zero_grad()

        # Accumulate total loss for reporting
        total_loss += (total_loss_cbf.item() + total_loss_action.item())
        avg_loss = total_loss / (istep + 1)

        # Saving the model at specific intervals
        if not os.path.exists('models'):
            os.makedirs('models')

        if (istep + 1) % config.SAVE_STEPS == 0:
            torch.save({
                'model_cbf_state_dict': model_cbf.state_dict(),
                'model_action_state_dict': model_action.state_dict(),
            }, os.path.join('models', f'model_iter_{istep + 1}.pth'))

        if (istep + 1) % config.DISPLAY_STEPS == 0:
            print('Step: {}, Loss: {:.4f}, Accuracy: {}'.format(
                istep + 1, avg_loss, np.array(count_accuracy(acc_lists_np))))
            # Resetting accumulators for next display interval
            total_loss = 0

    return avg_loss

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Starting training...")

    # Set random seed for reproducibility
    set_random_seeds(seed)

    model_cbf = core.NetworkCBF().to(device)
    model_action = core.NetworkAction().to(device)

    # Remove weight_decay from optimizer parameters
    optimizer_cbf = optim.Adam(model_cbf.parameters(), lr=config.LEARNING_RATE)
    optimizer_action = optim.Adam(model_action.parameters(), lr=config.LEARNING_RATE)

    if args.model_path is not None:
        if os.path.exists(args.model_path):
            print(f"Loading model from {args.model_path}...")
            checkpoint = torch.load(args.model_path, map_location=device)
            model_cbf.load_state_dict(checkpoint['model_cbf_state_dict'])
            model_action.load_state_dict(checkpoint['model_action_state_dict'])
        else:
            print(f"Model path {args.model_path} does not exist. Starting training from scratch.")

    dataset = AgentDataset(args.num_agents, config.DIST_MIN_THRES, num_samples=config.TRAIN_STEPS, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    avg_loss = train_epoch(args.num_agents, model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device)
    print(f'Final loss {avg_loss}')

if __name__ == '__main__':
    main()