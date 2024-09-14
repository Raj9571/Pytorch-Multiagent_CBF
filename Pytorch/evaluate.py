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

def get_next(s, g, a, model_cbf):
    x = s.unsqueeze(1) - s.unsqueeze(0)
    # h is the CBF value of shape [num_agents, TOP_K, 1], where TOP_K represents
    # the K nearest agents
    h, mask,_ = model_cbf(x=x, r=config.DIST_MIN_THRES)
    # a_res is delta a. when a does not satisfy the CBF conditions, we want to compute
    # a a_res such that a + a_res satisfies the CBF conditions
    a_res = torch.zeros_like(a, requires_grad=True)
    loop_count = torch.tensor(0, requires_grad=False)

    def opt_body(a_res, loop_count):
        # a loop of updating a_res
        # compute s_next under a + a_res
        dsdt = core.dynamics(s, a + a_res)
        s_next = s + dsdt * config.TIME_STEP
        x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)
        h_next, mask_next,_ = model_cbf(
            x=x_next, r=config.DIST_MIN_THRES)
        # deriv should be >= 0. if not, we update a_res by gradient descent
        deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h
        deriv = deriv * mask * mask_next
        error = torch.sum(torch.maximum(-deriv, torch.tensor(0.0)), axis=1)
        # print(error.shape)
        # print(a_res.shape)
        error = error.sum()
        # compute the gradient to update a_res
        error.backward(retain_graph=True)
        # error_gradient = torch.autograd.grad(error,a_res)
        optim = torch.optim.SGD([a_res], lr=0.01)
        optim.step()
        # print(f'a_res shape {a_res}')
        loop_count += 1
        return a_res, loop_count

    def opt_cond(a_res, loop_count):
        # update u_res for REFINE_LOOPS
        cond = loop_count < config.REFINE_LOOPS
        return cond
    
    with torch.no_grad():
        a_res.zero_()
        loop_count.zero_()

    while opt_cond(a_res, loop_count):
        a_res, loop_count = opt_body(a_res, loop_count)
        a_opt = a + a_res

    dsdt = core.dynamics(s, a_opt)
    s_next = s + dsdt * config.TIME_STEP
    x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)
    h_next, mask_next,_ = model_cbf(x=x_next, r=config.DIST_MIN_THRES)
    
    return h_next, s_next, a_opt, x_next

def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print('Accuracy: {}'.format(acc_list))

def render_init():
    fig = plt.figure(figsize=(9, 4))
    return fig

def main():
    args = parse_args()
    model_cbf_eval = core.NetworkCBF()
    model_action_eval = core.NetworkAction()

    checkpoint = torch.load("models/model_iter_4000.pth")
    model_cbf_eval.load_state_dict(checkpoint['model_cbf_state_dict'])
    model_action_eval.load_state_dict(checkpoint['model_action_state_dict'])
    
    model_cbf_eval.eval()
    model_action_eval.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        # randomly generate the initial conditions s_np_ori and the goal states g_np_ori
        s_np_ori, g_np_ori = core.generate_data(args.num_agents, config.DIST_MIN_THRES * 1.5)

        s_np, g_np = torch.tensor(s_np_ori, dtype=torch.float32).to(device), torch.tensor(g_np_ori, dtype=torch.float32).to(device)
        init_dist_errors.append(np.mean(np.linalg.norm(s_np[:, :2].detach().numpy() - g_np.cpu().numpy(), axis=1)))
        # store the trajectory for visualization
        s_np_ours = []
        s_np_lqr = []

        safety_ours = []
        safety_lqr = []
        j = 0
        # run INNER_LOOPS steps to reach the current goals
        for i in range(config.INNER_LOOPS):
            # compute the control input
            a_network = model_action_eval(s_np, g_np)
            dsdt = torch.cat([s_np[:, 2:], a_network], axis=1)
            # get h_next and s_next
            h_next, s_next, a_opt, x_next = get_next(s_np, g_np, a_network, model_cbf_eval)
            # simulate the system for one step
            (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
            h=h_next, s=s_next, r=config.DIST_MIN_THRES, 
            ttc=config.TIME_TO_COLLISION, eps=[0, 0])
            # loss_dang_deriv is for doth(s) + alpha h(s) >=0 for s in dangerous set
            # loss_safe_deriv is for doth(s) + alpha h(s) >=0 for s in safe set
            # loss_medium_deriv is for doth(s) + alpha h(s) >=0 for s not in the dangerous
            # or the safe set
            (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
                ) = core.loss_derivatives(s=s_next, a=a_opt, h=h_next, x=x_next, r=config.OBS_RADIUS, ttc=config.TIME_TO_COLLISION, alpha=config.ALPHA_CBF, time_step = config.TIME_STEP, dist_min_thres = config.DIST_MIN_THRES)
            # the distance between the u_opt and the nominal u
            loss_action = core.loss_actions(s_np, g_np, a_network, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

            loss_list = [loss_dang, loss_safe, loss_dang_deriv, loss_safe_deriv, loss_action]
            acc_list_np = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
            s_np = s_np + dsdt * config.TIME_STEP
            s_np_ours.append(s_np.detach().numpy())
            safety_ratio = 1 - np.mean(core.ttc_dangerous_mask_np(
                s_np.detach().numpy(), config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
            safety_ours.append(safety_ratio)
            safety_info.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio = np.mean(safety_ratio == 1)
            safety_ratios_epoch.append(safety_ratio)
            accuracy_lists.append(acc_list_np)

            if args.vis:
                # break if the agents are already very close to their goals
                if np.amax(
                    np.linalg.norm(s_np[:, :2].detach().numpy() - g_np.cpu().numpy(), axis=1)
                    ) < config.DIST_MIN_CHECK / 3:
                    time.sleep(1)
                    break
                # if the agents are very close to their goals, safely switch to LQR
                if np.mean(
                    np.linalg.norm(s_np[:, :2].detach().numpy() - g_np.cpu().numpy(), axis=1)
                    ) < config.DIST_MIN_CHECK / 2:
                    K = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
                    s_ref = np.concatenate([s_np[:, :2].detach().numpy() - g_np.cpu().numpy(), s_np[:, 2:].detach().numpy()], axis=1)
                    a_lqr = -s_ref.dot(K.T)
                    s_np = s_np + torch.tensor(np.concatenate(
                       [s_np[:, 2:].detach().cpu().numpy(), a_lqr], axis=1), dtype=torch.float32).to(device) * config.TIME_STEP

            else:
                if np.mean(
                    np.linalg.norm(s_np[:, :2].detach().numpy() - g_np.cpu().numpy(), axis=1)
                    ) < config.DIST_MIN_CHECK:
                    break
            j = j + 1
            print(f'inner loop {j}')

        dist_errors.append(np.mean(np.linalg.norm(s_np[:, :2].detach().numpy() - g_np.cpu().numpy(), axis=1)))
        safety_reward.append(np.mean(np.sum(np.concatenate(safety_info, axis=0) - 1, axis=0)))
        dist_reward.append(np.mean(
            (np.linalg.norm((s_np[:, :2] - g_np).detach().numpy(), axis=1) < 0.2).astype(np.float32) * 10))

        # run the simulation using LQR controller without considering collision
        s_np, g_np = np.copy(s_np_ori), np.copy(g_np_ori)
        for i in range(config.INNER_LOOPS):
            K = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
            s_ref = np.concatenate([s_np[:, :2] - g_np, s_np[:, 2:]], axis=1)
            a_lqr = -s_ref.dot(K.T)
            s_np = s_np + np.concatenate([s_np[:, 2:], a_lqr], axis=1) * config.TIME_STEP
            s_np_lqr.append(s_np)
            safety_ratio = 1 - np.mean(core.ttc_dangerous_mask_np(
                s_np, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
            safety_lqr.append(safety_ratio)
            safety_info_baseline.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio = np.mean(safety_ratio == 1)
            safety_ratios_epoch_lqr.append(safety_ratio)
            if np.mean(
                    np.linalg.norm(s_np[:, :2] - g_np, axis=1)
                    ) < config.DIST_MIN_CHECK / 3:
                    break

        safety_reward_baseline.append(np.mean(
            np.sum(np.concatenate(safety_info_baseline, axis=0) - 1, axis=0)))
        dist_reward_baseline.append(np.mean(
            (np.linalg.norm(s_np[:, :2] - g_np, axis=1) < 0.2).astype(np.float32) * 10))

        if args.vis:
            # visualize the trajectories
            vis_range = max(1, np.amax(np.abs(s_np_ori[:, :2])))
            agent_size = 100 / vis_range ** 2
            g_np = g_np / vis_range
            for j in range(max(len(s_np_ours), len(s_np_lqr))):
                plt.clf()
                
                plt.subplot(121)
                j_ours = min(j, len(s_np_ours)-1)
                s_np = s_np_ours[j_ours] / vis_range
                plt.scatter(s_np[:, 0], s_np[:, 1], 
                            color='darkorange', 
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np[:, 0], g_np[:, 1], 
                            color='deepskyblue', 
                            s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_ours[j_ours])
                plt.scatter(s_np[safety<1, 0], s_np[safety<1, 1], 
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

                plt.subplot(122)
                j_lqr = min(j, len(s_np_lqr)-1)
                s_np = s_np_lqr[j_lqr] / vis_range
                plt.scatter(s_np[:, 0], s_np[:, 1], 
                            color='darkorange', 
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np[:, 0], g_np[:, 1], 
                            color='deepskyblue', 
                            s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_lqr[j_lqr])
                plt.scatter(s_np[safety<1, 0], s_np[safety<1, 1], 
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

                fig.canvas.draw()
                plt.pause(0.01)
            plt.clf()
        end_time = time.time()
        print('Evaluation Step: {} | {}, Time: {:.4f}'.format(
            istep + 1, config.EVALUATE_STEPS, end_time - start_time))
    print('Distance Error (Final | Inititial): {:.4f} | {:.4f}'.format(
          np.mean(dist_errors), np.mean(init_dist_errors)))
    print('Mean Safety Ratio (Learning | LQR): {:.4f} | {:.4f}'.format(
          np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_lqr)))
    print('Reward Safety (Learning | LQR): {:.4f} | {:.4f}, Reward Distance: {:.4f} | {:.4f}'.format(
        np.mean(safety_reward), np.mean(safety_reward_baseline), 
        np.mean(dist_reward), np.mean(dist_reward_baseline)))

if __name__ == '__main__':
    main()
