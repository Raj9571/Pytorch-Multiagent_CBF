import sys
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import core
import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args

def get_next(s, g, a, model_cbf, device):
    """
    Refines the action using Control Barrier Functions (CBF) to ensure safety.

    Args:
        s (torch.Tensor): Current state [num_agents, 4]
        g (torch.Tensor): Goal states [num_agents, 2]
        a (torch.Tensor): Nominal action [num_agents, 2]
        model_cbf (nn.Module): CBF network model
        device (torch.device): Device to run computations on

    Returns:
        h_next (torch.Tensor): CBF values after refinement
        s_next (torch.Tensor): Next state after applying refined action
        a_opt (torch.Tensor): Optimized action after refinement
        x_next (torch.Tensor): Relative positions after refinement
        indices (torch.Tensor): Indices of top K nearest agents
    """
    # Compute relative positions
    x = s.unsqueeze(1) - s.unsqueeze(0)  # [num_agents, num_agents, 4]
    
    # Compute CBF values and masks, get indices
    h, mask, indices = model_cbf(x=x, r=config.DIST_MIN_THRES)
    # Detach h and mask to prevent autograd from tracking their history
    h = h.detach()
    mask = mask.detach()
    
    # Initialize a_res with requires_grad=True
    a_res = torch.zeros_like(a, requires_grad=True).to(device)
    
    # Initialize optimizer outside the loop
    optimizer = torch.optim.SGD([a_res], lr=config.REFINE_LEARNING_RATE)
    
    loop_count = 0  # Use Python integer for loop count

    while loop_count < config.REFINE_LOOPS:
        optimizer.zero_grad()
        
        # Compute dynamics with refined action
        dsdt = core.dynamics(s, a + a_res)  # [num_agents, 4]
        s_next = s + dsdt * config.TIME_STEP  # [num_agents, 4]
        
        # Compute next relative positions
        x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)  # [num_agents, num_agents, 4]
        
        # Compute CBF values and masks for next state, using indices
        h_next, mask_next, _ = model_cbf(x=x_next, r=config.DIST_MIN_THRES, indices=indices)
        
        # Compute derivative condition
        deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h  # [num_agents, TOP_K, 1]
        deriv = deriv * mask * mask_next  # Apply masks
        
        # Compute error: sum of negative derivations
        error = torch.sum(torch.relu(-deriv))
        
        # Backpropagate to compute gradients
        error.backward()
        
        # Perform optimization step
        optimizer.step()
        
        loop_count += 1

    # Compute optimized action
    a_opt = a + a_res.detach()  # Detach a_res to prevent further gradients
    
    # Compute next state with optimized action
    dsdt_opt = core.dynamics(s, a_opt)  # [num_agents, 4]
    s_next = s + dsdt_opt * config.TIME_STEP  # [num_agents, 4]
    
    # Compute relative positions for next state
    x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)  # [num_agents, num_agents, 4]
    
    # Compute CBF values and masks for next state, using indices
    h_next, mask_next, _ = model_cbf(x=x_next, r=config.DIST_MIN_THRES, indices=indices)
    
    return h_next, s_next, a_opt, x_next, indices

def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        valid_acc = acc_i[acc_i >= 0]
        if len(valid_acc) > 0:
            acc_list.append(np.mean(valid_acc))
        else:
            acc_list.append(0.0)
    print('Accuracy: {}'.format(acc_list))

def render_init():
    fig = plt.figure(figsize=(9, 4))
    return fig

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    model_cbf_eval = core.NetworkCBF().to(device)
    model_action_eval = core.NetworkAction().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_cbf_eval.load_state_dict(checkpoint['model_cbf_state_dict'])
    model_action_eval.load_state_dict(checkpoint['model_action_state_dict'])
    
    model_cbf_eval.eval()
    model_action_eval.eval()
    
    safety_ratios_epoch = []
    safety_ratios_epoch_lqr = []

    dist_errors = []
    init_dist_errors = []
    accuracy_lists = []

    safety_reward = []
    dist_reward = []
    safety_reward_baseline = []
    dist_reward_baseline = []

    if args.vis:
        plt.ion()
        plt.close()
        fig = render_init()

    for istep in range(config.EVALUATE_STEPS):
        start_time = time.time()
        
        safety_info = []
        safety_info_baseline = []
        
        # Randomly generate initial conditions and goal states
        s_np_ori, g_np_ori = core.generate_data(args.num_agents, config.DIST_MIN_THRES * 1.5)

        # Convert to tensors and move to device
        s_np = torch.tensor(s_np_ori, dtype=torch.float32).to(device)  # [num_agents, 4]
        g_np = torch.tensor(g_np_ori, dtype=torch.float32).to(device)  # [num_agents, 2]
        
        init_dist_errors.append(np.mean(np.linalg.norm(s_np_ori[:, :2] - g_np_ori, axis=1)))
        
        # Store trajectories for visualization
        s_np_ours = []
        s_np_lqr = []

        safety_ours = []
        safety_lqr = []
        
        # Run INNER_LOOPS steps to reach the goals
        for i in range(config.INNER_LOOPS):
            with torch.no_grad():
                # Compute nominal control action using the action network
                a_network = model_action_eval(s_np, g_np)  # [num_agents, 2]
            
            # Refine the action using CBF
            h_next, s_next, a_opt, x_next, indices = get_next(s_np, g_np, a_network, model_cbf_eval, device)
            
            # Compute losses and accuracies
            (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
                h=h_next, s=s_next, r=config.DIST_MIN_THRES, 
                ttc=config.TIME_TO_COLLISION, model_cbf=model_cbf_eval, indices=indices, eps=[0, 0])

            (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv) = core.loss_derivatives(
                s=s_next, a=a_opt, h=h_next, x=x_next, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION, 
                alpha=config.ALPHA_CBF, time_step=config.TIME_STEP, dist_min_thres=config.DIST_MIN_THRES, 
                model_cbf=model_cbf_eval, indices=indices)

            # Compute action loss
            loss_action = core.loss_actions(s_np, g_np, a_network, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

            acc_list_np = [acc_dang.item(), acc_safe.item(), acc_dang_deriv.item(), acc_safe_deriv.item()]

            # Update state with optimized action
            s_np = s_next  # Update state with optimized action
            s_np_ours.append(s_np.cpu().numpy())
            
            # Compute safety metrics
            safety_mask_np = core.ttc_dangerous_mask_np(
                s_np.cpu().numpy(), config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            safety_ratio = 1 - np.mean(safety_mask_np, axis=1)
            safety_ours.append(safety_ratio)
            safety_info.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio_mean = np.mean(safety_ratio == 1)
            safety_ratios_epoch.append(safety_ratio_mean)
            accuracy_lists.append(acc_list_np)

            # Check if agents are close to their goals
            if torch.mean(torch.norm(s_np[:, :2] - g_np, dim=1)).item() < config.DIST_MIN_CHECK:
                break

            if args.vis:
                # Visualization
                if torch.max(torch.norm(s_np[:, :2] - g_np, dim=1)).item() < config.DIST_MIN_CHECK / 3:
                    time.sleep(1)
                    break

        # Compute distance error
        dist_errors.append(np.mean(np.linalg.norm(s_np[:, :2].cpu().numpy() - g_np.cpu().numpy(), axis=1)))

        # Compute rewards
        safety_reward.append(np.mean(np.sum(np.concatenate(safety_info, axis=0) - 1, axis=0)))
        dist_reward.append(np.mean(
            (np.linalg.norm((s_np[:, :2] - g_np).cpu().numpy(), axis=1) < 0.2).astype(np.float32) * 10))

        # Run simulation using LQR controller without considering collision
        s_np_lqr_current = torch.tensor(s_np_ori, dtype=torch.float32).to(device)  # Reset to initial state

        for i in range(config.INNER_LOOPS):
            K = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
            s_ref = torch.cat([s_np_lqr_current[:, :2] - g_np, s_np_lqr_current[:, 2:]], dim=1).cpu().numpy()
            a_lqr = -s_ref.dot(K.T)  # [num_agents, 2]
            a_lqr_tensor = torch.tensor(a_lqr, dtype=torch.float32).to(device)
            dsdt_lqr = torch.cat([s_np_lqr_current[:, 2:], a_lqr_tensor], dim=1)  # [num_agents, 4]
            s_np_lqr_current = s_np_lqr_current + dsdt_lqr * config.TIME_STEP
            s_np_lqr.append(s_np_lqr_current.cpu().numpy())
            
            # Compute safety metrics for LQR
            safety_mask_np_lqr = core.ttc_dangerous_mask_np(
                s_np_lqr_current.cpu().numpy(), config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            safety_ratio_lqr = 1 - np.mean(safety_mask_np_lqr, axis=1)
            safety_lqr.append(safety_ratio_lqr)
            safety_info_baseline.append((safety_ratio_lqr == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio_mean_lqr = np.mean(safety_ratio_lqr == 1)
            safety_ratios_epoch_lqr.append(safety_ratio_mean_lqr)
            
            # Break if agents are very close to their goals
            if torch.mean(torch.norm(s_np_lqr_current[:, :2] - g_np, dim=1)).item() < config.DIST_MIN_CHECK / 3:
                break

        # Compute rewards for LQR
        safety_reward_baseline.append(np.mean(
            np.sum(np.concatenate(safety_info_baseline, axis=0) - 1, axis=0)))
        dist_reward_baseline.append(np.mean(
            (np.linalg.norm(s_np_lqr_current[:, :2].cpu().numpy() - g_np.cpu().numpy(), axis=1) < 0.2).astype(np.float32) * 10))

        if args.vis:
            # Visualize the trajectories
            vis_range = max(1, np.amax(np.abs(s_np_ori[:, :2])))
            agent_size = 100 / vis_range ** 2
            g_np_vis = g_np.cpu().numpy() / vis_range
            for j in range(max(len(s_np_ours), len(s_np_lqr))):
                plt.clf()
                
                # Visualization for "Ours" (Learning-based Controller)
                plt.subplot(121)
                j_ours = min(j, len(s_np_ours)-1)
                s_vis_ours = s_np_ours[j_ours] / vis_range
                plt.scatter(s_vis_ours[:, 0], s_vis_ours[:, 1], 
                            color='darkorange', 
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np_vis[:, 0], g_np_vis[:, 1], 
                            color='deepskyblue', 
                            s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_ours[j_ours])
                plt.scatter(s_vis_ours[safety < 1, 0], s_vis_ours[safety < 1, 1], 
                            color='red', 
                            s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('Ours: Safety Rate = {:.3f}'.format(
                    np.mean(safety_ratios_epoch)), fontsize=14)

                # Visualization for "LQR" (Baseline Controller)
                plt.subplot(122)
                j_lqr = min(j, len(s_np_lqr)-1)
                s_vis_lqr = s_np_lqr[j_lqr] / vis_range
                plt.scatter(s_vis_lqr[:, 0], s_vis_lqr[:, 1], 
                            color='darkorange', 
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np_vis[:, 0], g_np_vis[:, 1], 
                            color='deepskyblue', 
                            s=agent_size, label='Target', alpha=0.6)
                safety_lqr_current = np.squeeze(safety_lqr[j_lqr])
                plt.scatter(s_vis_lqr[safety_lqr_current < 1, 0], s_vis_lqr[safety_lqr_current < 1, 1], 
                            color='red', 
                            s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('LQR: Safety Rate = {:.3f}'.format(
                    np.mean(safety_ratios_epoch_lqr)), fontsize=14)

                plt.draw()
                plt.pause(0.01)
            plt.clf()

        end_time = time.time()
        print('Evaluation Step: {} / {}, Time: {:.4f}'.format(
            istep + 1, config.EVALUATE_STEPS, end_time - start_time))

    # After all evaluation steps, print metrics
    print_accuracy(accuracy_lists)
    print('Distance Error (Final | Initial): {:.4f} | {:.4f}'.format(
          np.mean(dist_errors), np.mean(init_dist_errors)))
    print('Mean Safety Ratio (Learning | LQR): {:.4f} | {:.4f}'.format(
          np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_lqr)))
    print('Reward Safety (Learning | LQR): {:.4f} | {:.4f}, Reward Distance: {:.4f} | {:.4f}'.format(
        np.mean(safety_reward), np.mean(safety_reward_baseline), 
        np.mean(dist_reward), np.mean(dist_reward_baseline)))

    if args.vis:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()