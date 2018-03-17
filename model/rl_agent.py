import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from .utils import *
from .optim import Optimizer
from .vec_env import DummyVecEnv, SubprocVecEnv
from functools import reduce
import operator
from itertools import count
import time
from .storage import RolloutStorage
from .visualize import visdom_plot
from gym import spaces


class RLAgent(object):

    # args: seed, cuda, gamma, tau, entropy_coef, batch_size, value_loss_coef

    def __init__(self, make_env_fn, args, device=None,
                 expt_dir='experiment', checkpoint_every=100, log_interval=100,
                 input_scale=1.0,
                 num_stack=1,
                 reward_scale=1.0,
                 reward_clip=False,
                 reward_min=-1,
                 reward_max=1,
                 hidden_state_shape=None,
                 process_string_func=None,
                 language_vocab_func=None,
                 pad_sym='<pad>',
                 model=None):
        self.args = args
        torch.manual_seed(self.args.seed)
        self.cuda = self.args.cuda and torch.cuda.is_available()
        if self.cuda:
            torch.cuda.manual_seed(self.args.seed)

        self.make_env_fn = make_env_fn
        self.device = device
        self.input_scale = input_scale
        self.num_stack = num_stack
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.process_string = process_string_func
        self.language_vocab_func = language_vocab_func
        self.transposeImage = True

        if self.process_string is None:
            self.process_string = lambda x: [0]
        else:
            self.pad_sym = pad_sym
            res = self.process_string(pad_sym)
            self.pad_id = res[0]
            assert self.pad_id == 0, 'The pad_id must be zero.'

        self.model = model
        if self.model is not None:
            if self.cuda:
                self.model.cuda(self.device)

            modelSize = 0
            for p in self.model.parameters():
                pSize = reduce(operator.mul, p.size(), 1)
                modelSize += pSize
            print('Model size: %d' % modelSize)

        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.log_interval = log_interval

        if hidden_state_shape is not None:
            self.hidden_state_shape = hidden_state_shape
        else:
            self.hidden_state_shape = (1,)

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)

    # Function for updating the current observation data (without the
    # instruction)
    def update_current_obs(self, obs, current_obs):
        shape_dim0 = self.env_shape_init[0]
        if self.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

        return current_obs

    def getCurrentLanguageVocab(self):
        if self.language_vocab_func is None:
            return []
        else:
            return self.language_vocab_func()

    def process_reward(self, rewards):
        def proc(reward):
            reward = reward / self.reward_scale
            if self.reward_clip:
                reward = min(max(reward, self.reward_min), self.reward_max)
            return reward
        return [proc(r) for r in rewards]

    def process_state(self, state, textdata_maxlen, current_obs):
        len_mission = np.zeros((len(state), 1), np.int32)
        mission = np.ones((len(state), textdata_maxlen),
                          np.int32) * self.pad_id

        for i in range(len(state)):
            if not isinstance(state[i], dict):
                state[i] = {'image': state[i], 'mission': ''}

            if self.transposeImage:
                tmpImg = np.transpose(state[i]['image'], (2, 0, 1))
            else:
                tmpImg = state[i]['image']

            state[i]['image'] = np.expand_dims(tmpImg, 0)
            curr_mission = self.process_string(state[i]['mission'])
            len_mission[i][0] = min(textdata_maxlen, len(curr_mission))
            mission[i, :len_mission[i][0]] = curr_mission[:len_mission[i][0]]

        img = np.concatenate([state[i]['image']
                              for i in range(len(state))], axis=0)
        img = img / self.input_scale

        img = torch.from_numpy(img).float()
        mission = torch.from_numpy(mission).long()
        len_mission = torch.from_numpy(len_mission).long()

        if self.cuda:
            img = img.cuda(self.device)
            mission = mission.cuda(self.device)
            len_mission = len_mission.cuda(self.device)

        img = self.update_current_obs(img, current_obs)

        return State(img, mission, len_mission)

    def state_2_variable(self, state, volatile=False):
        return State(
            Variable(state.image, volatile=volatile),
            Variable(state.mission, volatile=volatile),
            Variable(state.missionlen, volatile=volatile)
        )

    def a2cProcess(self, rollouts):

        textdata_maxlen = rollouts.missions.size(2)

        # evaluate chosen actions
        tmp = self.model.evaluate_actions(
            self.state_2_variable(
                State(
                    rollouts.observations[:-1].view(-1, *self.obs_shape),
                    rollouts.missions[:-1].view(-1, textdata_maxlen),
                    rollouts.missionlen[:-1].view(-1, 1)
                )
            ),
            Variable(rollouts.hiddenStates[:-1].view(
                -1, *self.hidden_state_shape)),
            Variable(rollouts.masks[:-1].view(-1, 1)),
            Variable(rollouts.actions.view(-1, self.action_shape))
        )
        values, action_log_probs, dist_entropy, hiddenStates = tmp

        # reshape the values and the log_prob
        values = values.view(-1, self.args.num_processes, 1)
        action_log_probs = action_log_probs.view(
            -1, self.args.num_processes, 1)

        # computations of the value_loss
        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        # computations of the policy_loss
        policy_loss = -(Variable(advantages.data) * action_log_probs).mean()
        entropy_loss = -dist_entropy * self.args.entropy_coef

        # computations of the total loss
        total_loss = (
            entropy_loss +
            policy_loss +
            self.args.value_loss_coef * value_loss
        )

        # optimization step
        self.optimizer.zero_grad()

        # Back-propagation
        total_loss.backward()

        # Apply updates
        self.optimizer.step()

        lossComponents = {
            'policy_loss': policy_loss.cpu().data.numpy()[0],
            'value_loss': value_loss.cpu().data.numpy()[0],
            'dist_entropy': dist_entropy.cpu().data.numpy()[0],
        }

        return total_loss.cpu().data.numpy()[0], lossComponents

    def ppoProcess(self, rollouts):

        te = 1e-5
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + te)

        # for statistics computations
        nupdates = 0
        loss_mean = 0.0
        policy_loss_mean = 0.0
        value_loss_mean = 0.0
        dist_entropy_mean = 0.0

        for e in range(self.args.ppo_epoch):
            if self.model.hasRecurrence():
                data_generator = rollouts.recurrent_generator(
                    advantages,
                    min(self.args.ppo_num_mini_batch, self.args.num_processes),
                    self.device
                )
            else:
                data_generator = rollouts.forward_generator(
                    advantages,
                    # min(self.args.ppo_num_mini_batch, self.args.num_processes),
                    self.args.ppo_num_mini_batch,
                    self.device
                )

            for sample in data_generator:

                observations_batch, mission_batch, missionlen_batch, \
                    hiddenStates_batch, actions_batch, return_batch, \
                    masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                tmp = self.model.evaluate_actions(
                    self.state_2_variable(
                        State(
                            observations_batch,
                            mission_batch,
                            missionlen_batch
                        )
                    ),
                    Variable(hiddenStates_batch),
                    Variable(masks_batch),
                    Variable(actions_batch)
                )
                values, action_log_probs, dist_entropy, hiddenStates = tmp

                adv_targ = Variable(adv_targ)
                ratio = torch.exp(
                    action_log_probs - Variable(old_action_log_probs_batch)
                )
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.args.ppo_clip_param,
                    1.0 + self.args.ppo_clip_param
                ) * adv_targ

                # PPO's pessimistic surrogate (L^CLIP)
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -dist_entropy * self.args.entropy_coef

                # Value loss
                value_loss = (Variable(return_batch) - values).pow(2).mean()

                # total loss
                total_loss = value_loss + policy_loss + entropy_loss

                # optimization step
                self.optimizer.zero_grad()

                # Back-propagation
                total_loss.backward()

                # Apply updates
                self.optimizer.step()

                # update statistics
                nupdates += 1
                loss_mean += total_loss.cpu().data.numpy()[0]
                policy_loss_mean += policy_loss.cpu().data.numpy()[0]
                value_loss_mean += value_loss.cpu().data.numpy()[0]
                dist_entropy_mean += dist_entropy.cpu().data.numpy()[0]

        lossComponents = {
            'policy_loss': policy_loss_mean / max(1, nupdates),
            'value_loss': value_loss_mean / max(1, nupdates),
            'dist_entropy': dist_entropy_mean / max(1, nupdates)
        }

        return loss_mean / max(1, nupdates), lossComponents

    def optimize_model(self, rollouts):

        if self.args.algo == 'a2c':
            total_loss, lossComponents = self.a2cProcess(rollouts)
        elif self.args.algo == 'ppo':
            total_loss, lossComponents = self.ppoProcess(rollouts)
        else:
            raise ValueError('unknown algo: [' + self.args.algo + ']')

        return total_loss, lossComponents

    def _trainEpisodes(self, max_iters, start_epoch, start_step,
                       textdata_maxlen, max_fwd_steps=None, path='.'):

        step = start_step
        updatecount = 0
        epoch_loss_total = 0

        # define the checkout index
        chekoutindex = int(step / self.checkpoint_every) + 1  # 1
        current_episode = start_epoch

        if max_fwd_steps is None:
            max_fwd_steps = int(
                max_iters - step) // self.args.num_processes // 500

        # define the storange memory for all the rollouts
        rollouts = RolloutStorage(
            max_fwd_steps,
            self.args.num_processes,
            self.obs_shape,
            textdata_maxlen,
            self.env.action_space,
            self.hidden_state_shape
        )

        # data structure to collect the current observation
        # especially if numstack > 1
        current_obs = torch.zeros(
            self.args.num_processes, *self.obs_shape)

        if self.cuda:
            rollouts = rollouts.cuda(self.device)
            current_obs = current_obs.cuda(self.device)

        # rest the environment and get the first observation/state data
        state = self.env.reset()
        state = self.process_state(
            state, textdata_maxlen, current_obs)

        # update current observation
        current_obs.copy_(state.image)

        # copy the first observation (image & mission) in the memory
        rollouts.observations[0].copy_(state.image)
        rollouts.missions[0].copy_(state.mission)
        rollouts.missionlen[0].copy_(state.missionlen)

        # These variables are used to compute average reward for all processes
        episode_rewards = torch.zeros([self.args.num_processes, 1])
        final_rewards = torch.zeros([self.args.num_processes, 1])

        self.model.train()

        start = time.time()

        while step < max_iters:

            local_elapsed_steps = 0

            for ep_len in range(max_fwd_steps):

                # update steps counters
                local_elapsed_steps += self.args.num_processes

                # Sample actions
                value, action, action_log_prob, hiddenStates = self.model.act(
                    self.state_2_variable(
                        State(
                            rollouts.observations[ep_len],
                            rollouts.missions[ep_len],
                            rollouts.missionlen[ep_len]
                        ),
                        volatile=True
                    ),
                    Variable(rollouts.hiddenStates[ep_len], volatile=True),
                    Variable(rollouts.masks[ep_len], volatile=True)
                )

                # get the action on cpu
                cpu_actions = action.data.squeeze(1).cpu().numpy()

                # Perform the action on the env (Obser reward and next obs)
                state, reward, done, info = self.env.step(cpu_actions)
                reward = self.process_reward(reward)
                reward = torch.from_numpy(
                    np.expand_dims(np.stack(reward), 1)
                ).float()
                episode_rewards += reward

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]
                )
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                # clean the history of observations
                if self.args.cuda:
                    masks = masks.cuda()
                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                elif current_obs.dim() == 3:
                    current_obs *= masks.unsqueeze(2)
                else:
                    current_obs *= masks

                # preprocess state data
                state = self.process_state(
                    state, textdata_maxlen, current_obs)

                # update current observation
                current_obs.copy_(state.image)

                # save the collected information
                rollouts.insert(
                    ep_len,
                    state.image,
                    state.mission,
                    state.missionlen,
                    hiddenStates.data,
                    action.data,
                    action_log_prob.data,
                    value.data,
                    reward,
                    masks
                )

            next_value, _, _, _ = self.model.act(
                self.state_2_variable(
                    State(
                        rollouts.observations[-1],
                        rollouts.missions[-1],
                        rollouts.missionlen[-1]
                    ),
                    volatile=True
                ),
                Variable(rollouts.hiddenStates[-1], volatile=True),
                Variable(rollouts.masks[-1], volatile=True)
            )
            next_value = next_value.data

            rollouts.compute_returns(
                next_value,
                self.args.use_gae,
                self.args.gamma,
                self.args.tau
            )

            # compute the loss
            loss, lossComponents = self.optimize_model(rollouts)

            # update the global couter
            step += local_elapsed_steps
            current_episode += self.args.num_processes
            epoch_loss_total += loss
            updatecount += 1
            epoch_loss_avg = epoch_loss_total / updatecount

            # update the rollout storage memory for net run
            rollouts.after_update()

            # update optimizer statistics
            self.optimizer.update(epoch_loss_avg, updatecount)

            # Checkpoint
            if step >= chekoutindex * self.checkpoint_every or step >= max_iters:
                Checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=current_episode,
                    step=step,
                    vocab=self.getCurrentLanguageVocab()).save(
                    self.expt_dir)
                chekoutindex += 1

            if ((updatecount - 1) % self.log_interval == 0):
                end = time.time()
                total_num_steps = (
                    updatecount * self.args.num_processes * max_fwd_steps
                )
                print(
                    "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                    format(
                        updatecount - 1,
                        total_num_steps,
                        int(total_num_steps / (end - start)),
                        final_rewards.mean(),
                        final_rewards.median(),
                        final_rewards.min(),
                        final_rewards.max(),
                        lossComponents['dist_entropy'],
                        lossComponents['value_loss'],
                        lossComponents['policy_loss']
                    )
                )

            if ((self.args.vis) and (
                    (updatecount - 1) % self.args.vis_interval == 0)):
                total_num_steps = (
                    updatecount * self.args.num_processes * max_fwd_steps
                )
                win = visdom_plot(
                    total_num_steps,
                    final_rewards.mean()
                )

    def create_envs(self, num_envs=1):
        env = [self.make_env_fn(self.args.env_name, self.args.seed, i)
               for i in range(num_envs)]
        if len(env) == 1:
            env = DummyVecEnv(env)
        else:
            env = SubprocVecEnv(env)  # DummyVecEnv(env)  #

        if isinstance(env.observation_space, spaces.Dict):
            obs_space = env.observation_space.__dict__['spaces']['image']
        else:
            obs_space = env.observation_space

        self.obs_shape = obs_space.shape
        if self.transposeImage:
            self.obs_shape = (
                self.obs_shape[2],
                self.obs_shape[0],
                self.obs_shape[1]
            )
        self.env_shape_init = self.obs_shape
        self.obs_shape = (self.obs_shape[0] * self.args.num_stack, *self.obs_shape[1:])
        self.obs_numel = reduce(operator.mul, self.obs_shape, 1)
        if env.action_space.__class__.__name__ == "Discrete":
            self.action_shape = 1
        else:
            self.action_shape = env.action_space.shape[0]

        return env

    def train(
            self, max_iters, textdata_maxlen=50, num_steps=None,
            resume=False, optimizer=None, path='.'):
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(
                self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            self.model = resume_checkpoint.model
            if self.cuda and torch.cuda.is_available():
                self.model.cuda(self.device)
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            self.optimizer.optimizer = resume_optim.__class__(
                self.model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
            self.hidden_state_shape = self.model.hidden_state_shape()
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(
                    optim.Adam(
                        self.model.parameters(), lr=self.args.lr),
                    max_grad_norm=self.args.max_grad_norm)
            else:
                optimizer = Optimizer(
                    optimizer,
                    max_grad_norm=self.args.max_grad_norm)
            self.optimizer = optimizer

        self.env = self.create_envs(self.args.num_processes)

        self._trainEpisodes(
            max_iters, start_epoch, step,
            textdata_maxlen, num_steps,
            path=path
        )

        return self.model

    def test(self, max_iters=None, textdata_maxlen=50, render=False):

        self.env = self.env = self.create_envs(1)

        # get the number of env
        num_envs = 1

        # data structure to collect the current observation
        # especially if numstack > 1
        current_obs = torch.zeros(num_envs, *self.obs_shape)
        masks = torch.ones(num_envs, 1)

        if self.cuda:
            current_obs = current_obs.cuda(self.device)
            masks = masks.cuda()

        # These variables are used to compute average reward for all processes
        episode_reward = torch.zeros(num_envs, 1)
        final_rewards = torch.zeros(num_envs, 1)

        # reset the environment and get the first observation/state data
        state = self.env.reset()
        state = self.process_state(
            state, textdata_maxlen, current_obs)

        # update current observation
        current_obs.copy_(state.image)

        # hidden state
        hiddenStates = None

        self.model.eval()

        step = 0
        updatecount = 0

        start_time = time.time()

        while True:
            step += num_envs
            updatecount += 1

            value, action, action_log_prob, hiddenStates = self.model.act(
                self.state_2_variable(state, volatile=True),
                hiddenStates,
                Variable(masks, volatile=True),
                deterministic=True
            )

            # get the action on cpu
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Perform the action on the env (Obser reward and next obs)
            state, reward, done, info = self.env.step(cpu_actions)
            reward = self.process_reward(reward)
            reward = torch.from_numpy(
                np.expand_dims(np.stack(reward), 1)
            ).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]
            )
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if render and isinstance(self.env, DummyVecEnv):
                self.env.envs[0].render()
                time.sleep(0.1)

            # clean the history of observations
            if self.args.cuda:
                masks = masks.cuda()
            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            elif current_obs.dim() == 3:
                current_obs *= masks.unsqueeze(2)
            else:
                current_obs *= masks

            # preprocess state data
            state = self.process_state(
                state, textdata_maxlen, current_obs)

            # update current observation
            current_obs.copy_(state.image)

            if ((updatecount - 1) % self.log_interval == 0):
                end = time.time()
                total_num_steps = updatecount * num_envs

                print(
                    "Time {}, Updates {}, steps {}, FPS {}, mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}".
                    format(
                        time.strftime(
                            "%Hh %Mm %Ss", time.gmtime(end - start_time)),
                        updatecount - 1,
                        step,
                        total_num_steps / (end - start_time),
                        final_rewards.mean(),
                        final_rewards.median(),
                        final_rewards.min(),
                        final_rewards.max()
                    )
                )

                time.sleep(1)

            if (max_iters is not None) and (step >= max_iters):
                break

        return self.model
