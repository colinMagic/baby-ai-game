import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):

    def __init__(
            self,
            num_steps,
            num_processes,
            obs_shape,
            textdata_maxlen,
            action_space,
            hidden_state_shape):
        self.observations = torch.zeros(
            num_steps + 1, num_processes, *obs_shape)
        self.missions = torch.zeros(
            num_steps + 1, num_processes, textdata_maxlen).long()
        self.missionlen = torch.zeros(
            num_steps + 1, num_processes, 1).long()
        self.hiddenStates = torch.zeros(
            num_steps + 1, num_processes, *hidden_state_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

    def cuda(self, device=None):
        self.observations = self.observations.cuda(device)
        self.missions = self.missions.cuda(device)
        self.missionlen = self.missionlen.cuda(device)
        self.hiddenStates = self.hiddenStates.cuda(device)
        self.rewards = self.rewards.cuda(device)
        self.value_preds = self.value_preds.cuda(device)
        self.returns = self.returns.cuda(device)
        self.action_log_probs = self.action_log_probs.cuda(device)
        self.actions = self.actions.cuda(device)
        self.masks = self.masks.cuda(device)
        return self

    def insert(
            self,
            step,
            current_obs,
            mission,
            mission_len,
            hiddenState,
            action,
            action_log_prob,
            value_pred,
            reward,
            mask):
        self.observations[step + 1].copy_(current_obs)
        self.missions[step + 1].copy_(mission)
        self.missionlen[step + 1].copy_(mission_len)
        self.hiddenStates[step + 1].copy_(hiddenState)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.missions[0].copy_(self.missions[-1])
        self.missionlen[0].copy_(self.missionlen[-1])
        self.hiddenStates[0].copy_(self.hiddenStates[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def forward_generator(self, advantages, num_mini_batch, device=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False
        )
        for indices in sampler:
            indices = torch.LongTensor(indices)

            if advantages.is_cuda:
                indices = indices.cuda(device)

            observations_batch = self.observations[
                :-1].view(-1, *self.observations.size()[2:])[indices]
            mission_batch = self.missions[
                :-1].view(-1, *self.missions.size()[2:])[indices]
            missionlen_batch = self.missionlen[
                :-1].view(-1, *self.missionlen.size()[2:])[indices]
            hiddenStates_batch = self.hiddenStates[
                :-1].view(-1, *self.hiddenStates.size()[2:])[indices]
            actions_batch = self.actions.view(
                -1, *self.actions.size()[2:])[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(
                -1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, mission_batch, missionlen_batch, \
                hiddenStates_batch, actions_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch, device=None):
        num_processes = self.rewards.size(1)
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = []
            mission_batch = []
            missionlen_batch = []
            hiddenStates_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                observations_batch.append(self.observations[:-1, ind])
                mission_batch.append(self.missions[:-1, ind])
                missionlen_batch.append(self.missionlen[:-1, ind])
                hiddenStates_batch.append(self.hiddenStates[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            observations_batch = torch.cat(observations_batch, 0)
            mission_batch = torch.cat(mission_batch, 0)
            missionlen_batch = torch.cat(missionlen_batch, 0)
            hiddenStates_batch = torch.cat(hiddenStates_batch, 0)
            actions_batch = torch.cat(actions_batch, 0)
            return_batch = torch.cat(return_batch, 0)
            masks_batch = torch.cat(masks_batch, 0)
            old_action_log_probs_batch = torch.cat(
                old_action_log_probs_batch, 0)
            adv_targ = torch.cat(adv_targ, 0)

            yield observations_batch, mission_batch, missionlen_batch, \
                hiddenStates_batch, actions_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ
