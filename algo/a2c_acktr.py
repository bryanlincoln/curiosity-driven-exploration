import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 use_curiosity=False,
                 fwd_model=None,
                 inv_model=None,
                 curiosity_beta=0.2,
                 curiosity_lambda=0.1):

        self.actor_critic = actor_critic
        self.acktr = acktr
        self.use_curiosity = use_curiosity
        if self.use_curiosity:
            self.fwd_model = fwd_model
            self.inv_model = inv_model
            self.curiosity_beta = curiosity_beta
            self.curiosity_lambda = curiosity_lambda

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        
        if not self.use_curiosity:
            if acktr:
                self.optimizer = KFACOptimizer(actor_critic)
            else:
                self.optimizer = optim.RMSprop(
                    actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        else:
            if acktr:
                raise NotImplementedError
                self.optimizer = KFACOptimizer(actor_critic)
            else:
                self.optimizer = optim.RMSprop(
                        [{'params': actor_critic.parameters()},
                         {'params': fwd_model.parameters()},
                         {'params': inv_model.parameters()}], lr, eps=eps, alpha=alpha)

    def update(self, rollouts, device=None):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        
        if not self.use_curiosity:
            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))
        else:
            values, action_log_probs, dist_entropy, _, actor_features = self.actor_critic.evaluate_actions_curiosity(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape),
                self.fwd_model, self.inv_model)
            
            # Compute fwd_preds, inv_preds
            actions_onehot = torch.Tensor((num_steps, num_processes, rollouts.n_actions)).to(device)
            actions_onehot.scatter_(2, rollouts.actions, 1)
            states = actor_features.view(num_steps, num_processes, -1)[:-1]
            next_states = actor_features.view(num_steps, num_processes, -1)[1:]
            states = states.view(num_steps*num_processes, -1)
            next_states = next_states.view(num_steps*num_processes, -1)
            actions_onehot = actions_onehot[:-1].view(num_steps*num_processes, -1)
            # ================= Forward loss ===============
            # actor_features -> num_steps*num_processes x 512 
            # actions_onehot -> num_steps*num_processes x 512
            fwd_preds = self.fwd_model(states, actions_onehot)
            fwd_loss = 0.5*torch.mean(((fwd_preds - next_states) ** 2).sum(dim=1))
            # ================= Inverse loss ===============
            # Inverse loss by pairing (s0, s1)->a0, (s1, s2)->a1, ..., (sN-2, sN-1)->aN-2
            inv_preds = self.inv_model(states, next_states)
            inv_loss = F.cross_entropy(inv_preds, rollouts.actions[:-1].view(-1).long())

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        if not self.use_curiosity:
            (value_loss * self.value_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()
        else:
            (self.curiosity_lambda*(value_loss * self.value_loss_coef + 
                action_loss - dist_entropy * self.entropy_coef) + 
             self.curiosity_beta*fwd_loss + 
             (1-self.curiosity_beta)*inv_loss).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            if self.use_curiosity:
                nn.utils.clip_grad_norm_(self.fwd_model.parameters(),
                                         self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.inv_model.parameters(),
                                         self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
