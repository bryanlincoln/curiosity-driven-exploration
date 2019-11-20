import os
import copy
import glob
import time
import types
from collections import deque

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import algo
import tensorboardX

from envs import make_vec_envs
from utils import get_env_mean_std
from model import Policy, ForwardModel, InverseModel
from storage import RolloutStorage
from arguments import get_args
from visualize import visdom_plot
from tensorboardX import SummaryWriter

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    pass
#    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
#    for f in files:
#        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    pass
#    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
#    for f in files:
#        os.remove(f)


def main():

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    tbwriter = SummaryWriter(log_dir=args.save_dir)

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device, False)

    args.obs_mean, args.obs_std = get_env_mean_std(args.env_name, args.seed)

    actor_critic = Policy(envs.observation_space, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy, 'obs_mean': args.obs_mean, 'obs_std': args.obs_std})

    if args.use_curiosity:
        fwd_model = ForwardModel(
            envs.action_space, state_size=512, hidden_size=256)
        inv_model = InverseModel(
            envs.action_space, state_size=512, hidden_size=256)
        fwd_model.to(device)
        inv_model.to(device)
    else:
        fwd_model = None
        inv_model = None

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               acktr=False, norm_adv=args.norm_adv,
                               use_curiosity=args.use_curiosity,
                               fwd_model=fwd_model, inv_model=inv_model,
                               curiosity_beta=args.curiosity_beta,
                               curiosity_lambda=args.curiosity_lambda)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm,
                         use_curiosity=args.use_curiosity,
                         fwd_model=fwd_model, inv_model=inv_model,
                         curiosity_beta=args.curiosity_beta,
                         curiosity_lambda=args.curiosity_lambda)
    elif args.algo == 'acktr':
        if args.use_curiosity:
            raise NotImplementedError
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    # TODO 'shape' call breaks on REAL env
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, norm_rew=args.norm_rew)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            # print(rollouts.recurrent_hidden_states[step])
            # print(type(rollouts.recurrent_hidden_states[step]))
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, actor_features = actor_critic.act_curiosity(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            reward = reward.to(device)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]).to(device)

            if args.use_curiosity:
                with torch.no_grad():
                    next_actor_features = actor_critic.get_features(
                        obs, recurrent_hidden_states, masks).detach()

                # Augment reward with curiosity rewards
                if envs.action_space.__class__.__name__ == "Discrete":
                    action_curiosity = torch.zeros(
                        args.num_processes, envs.action_space.n, device=device)
                    action_curiosity.scatter_(1, action.view(-1, 1).long(), 1)
                elif envs.action_space.__class__.__name__ == "Box":
                    action_curiosity = action

                with torch.no_grad():
                    pred_actor_features = fwd_model(
                        actor_features, action_curiosity).detach()
                    curiosity_rewards = 0.5 * \
                        torch.mean(F.mse_loss(
                            pred_actor_features, next_actor_features, reduction='none'), dim=1).view(-1, 1)
                reward = reward + args.curiosity_eta * curiosity_rewards

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau)

        if not args.use_curiosity:
            value_loss, action_loss, dist_entropy = agent.update(
                rollouts, device=device)
        else:
            value_loss, action_loss, dist_entropy, fwd_loss, inv_loss = agent.update(
                rollouts, device=device)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

            torch.save(save_model, os.path.join(
                save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                         len(episode_rewards),
                         np.mean(episode_rewards),
                         np.median(episode_rewards),
                         np.min(episode_rewards),
                         np.max(episode_rewards), dist_entropy,
                         value_loss, action_loss))

            tbwriter.add_scalar('mean_reward', np.mean(
                episode_rewards), total_num_steps)
            tbwriter.add_scalar('median_reward', np.median(
                episode_rewards), total_num_steps)
            tbwriter.add_scalar('dist_entropy', dist_entropy, total_num_steps)
            tbwriter.add_scalar('value_loss', value_loss, total_num_steps)
            tbwriter.add_scalar('action_loss', action_loss, total_num_steps)

            if args.use_curiosity:
                print("fwd loss: {:.5f}, inv loss: {:.5f}".format(
                    fwd_loss, inv_loss))
                tbwriter.add_scalar('fwd_loss', fwd_loss, total_num_steps)
                tbwriter.add_scalar('inv_loss', inv_loss, total_num_steps)

        if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                      args.gamma, eval_log_dir, args.add_timestep, device, True)

            if eval_envs.venv.__class__.__name__ == "VecNormalize":
                eval_envs.venv.ob_rms = envs.venv.ob_rms

                # An ugly hack to remove updates
                def _obfilt(self, obs):
                    if self.ob_rms:
                        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(
                            self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                        return obs
                    else:
                        return obs

                eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                       actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                  format(len(eval_episode_rewards),
                         np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass


if __name__ == "__main__":
    main()
