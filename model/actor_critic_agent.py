import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple
from .utils import *
from .model import Model
from .optim import Optimizer
from functools import reduce
import operator
from itertools import count
import time


class ActorCritic_Agent(object):

    # args: seed, cuda, gamma, tau, entropy_coef, batch_size, value_loss_coef

    def __init__(self, env, args, device=None,
                 expt_dir='experiment', checkpoint_every=100, log_interval=100,
                 usememory=True, memory_capacity=20000, num_frame_rp=3,
                 memory_start=200,
                 input_scale=1.0,
                 reward_scale=1.0,
                 reward_clip=False,
                 reward_min=-1,
                 reward_max=1,
                 pad_sym='<pad>',
                 eos_sym='<eos>',
                 unk_sym='<unk>',
                 vocab_dict=None,
                 target_keywords=None,
                 visual_target_func_ind=None,
                 model_args=None):
        self.args = args
        torch.manual_seed(self.args.seed)
        self.env = env
        self.model_args = model_args
        self.cuda = self.args.cuda and torch.cuda.is_available()
        self.device = device
        self.input_scale = input_scale
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.pad_sym = pad_sym
        self.eos_sym = eos_sym
        self.unk_sym = unk_sym
        self.vtcFuncInd = visual_target_func_ind
        self.vocab = vocab_dict
        self.target_keywords = target_keywords
        if self.vocab is None:
            self.vocab = {self.pad_sym: 0, self.unk_sym: 1, self.eos_sym: 2}

        self.pad_id = self.vocab[self.pad_sym]
        self.unk_id = self.vocab[self.unk_sym]
        self.eos_id = self.vocab[self.eos_sym]
        assert self.pad_id == 0, 'The pad_id must be zero.'

        self.lpweight = torch.ones(len(self.vocab))
        self.lpweight[self.pad_id] = 0
        self.lpweight[self.eos_id] = 0
        if self.target_keywords is not None and (
                len(self.target_keywords) > 0):
            self.target_keywords = set([
                self.vocab[a.lower().strip()] for a in self.target_keywords
            ])
            for k, v in self.vocab.items():
                if v not in self.target_keywords:
                    self.lpweight[v] = 0

        if hasattr(self.env, 'seed'):
            self.env.seed(self.args.seed)
        self.model = None
        if model_args is not None:
            self.model = Model(**model_args)
            if self.cuda:
                self.model.cuda(device)
                self.lpweight.cuda(device)

        if self.model is not None:
            modelSize = 0
            for p in self.model.parameters():
                pSize = reduce(operator.mul, p.size(), 1)
                modelSize += pSize
            print('Model size: %d' % modelSize)

        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.log_interval = log_interval

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)

        self.usememory = usememory
        self.memory = None

        if self.usememory:
            self.memory_capacity = memory_capacity
            self.num_frame_rp = num_frame_rp
            self.memory_start = memory_start
            self.memory = ReplayMemory(memory_capacity, num_frame_rp)

    def process_reward(self, reward):
        reward = reward / self.reward_scale
        if self.reward_clip:
            reward = min(max(reward, self.reward_min), self.reward_max)
        return reward

    def process_string(self, text):
        if (text is None) or (text == ''):
            return [self.eos_id]
        else:
            words = text.strip().lower().split()
            word_ids = []
            for i in range(len(words)):
                key = words[i].strip()
                if key == '':
                    continue
                elif key in self.vocab:
                    word_ids.append(self.vocab[key])
                else:
                    word_ids.append(self.unk_id)
            word_ids.append(self.eos_id)
            return word_ids

    def process_state(self, state, volatile=False):
        if not isinstance(state, dict):
            state = {'image': state, 'mission': '', 'advice': ''}

        state['mission'] = self.process_string(state['mission'])
        state['advice'] = self.process_string(state['advice'])

        img = np.expand_dims(np.transpose(state['image'], (2, 0, 1)), 0)
        img = img / self.input_scale
        order = np.expand_dims((state['mission']), 0)
        advice = np.expand_dims((state['advice']), 0)

        if self.cuda:
            img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
            order = torch.from_numpy(order).type(torch.cuda.LongTensor)
            advice = torch.from_numpy(advice).type(torch.cuda.LongTensor)
        else:
            img = torch.from_numpy(img).type(torch.FloatTensor)
            order = torch.from_numpy(order).type(torch.LongTensor)
            advice = torch.from_numpy(advice).type(torch.LongTensor)

        return State(
            Variable(img, volatile=volatile),
            Variable(order, volatile=volatile),
            Variable(advice, volatile=volatile)
        )

    def fill_memory_replay(self, init_state):
        if self.memory is None:
            return init_state

        state = init_state
        inside = False
        while len(self.memory) < self.memory_start:
            inside = True

            # to avoid building the graph, set the volatility to True
            # before going through the model
            logit, value = self.model(state)

            # Calculate entropy from action probability distribution
            action_probs = F.softmax(logit, dim=1)
            m = Categorical(action_probs)
            action = m.sample()
            action = action.data.cpu().numpy()[0]

            nextstate, reward, done, _ = self.env.step(action)
            vtcInd = 0
            if not self.vtcFuncInd is None:
                img = nextstate if not isinstance(
                    nextstate, dict) else nextstate['image']
                vtcInd = int(self.vtcFuncInd(self.env, img))
            next_state = self.process_state(nextstate)
            reward = self.process_reward(reward)

            frame = ReplayFrame(
                state, logit.detach(), next_state,
                reward, value.detach(), done, vtcInd
            )
            self.memory.push(frame)
            if not done:
                state = next_state
            else:
                state = self.env.reset()
                state = self.process_state(state)
                self.model.reset_hidden_states()
                self.memory.reset_rp_frame_index()

        if inside:
            print('The memory has been pre-filled.')

        return state

    def compute_reversed_reward(self, R_vr, rewards):
        output = [0.0] * len(rewards)
        for i in reversed(range(len(rewards))):
            R_vr = self.args.gamma * R_vr + rewards[i]
            output[i] = R_vr
        return output

    def identify_lp_target_keywords(self, mission):
        mis = mission.view(-1)
        out = []
        for i in range(mis.numel()):
            if mis.data[i] in self.target_keywords:
                out.append(mis.data[i])
        return out

    def optimize_model(self, values, log_probs, rewards, entropies):

        R = values[-1]
        gae = torch.zeros(1, 1).type_as(R.data)

        # Base A3C Loss
        policy_loss, value_loss = 0, 0

        # Performing update
        for i in reversed(range(len(rewards))):
            # Value function loss
            R = self.args.gamma * R + rewards[i]
            value_loss = value_loss + 0.5 * (R - values[i]).pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + self.args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * self.args.gamma * self.args.tau + delta_t

            # Computing policy loss
            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - self.args.entropy_coef * entropies[i]

        # Auxiliary loss
        language_prediction_loss = 0
        tae_loss = 0
        reward_prediction_loss = 0
        value_replay_loss = 0
        vtc_loss = 0  # visual_target_classification_loss

        if not self.memory is None:
            # Non-skewed sampling from experience buffer
            auxiliary_samples = self.memory.sample(self.args.batch_size)
            auxiliary_batch = ReplayFrame(*zip(*auxiliary_samples))

            # TAE Loss
            # print('tae pred')
            if self.model.tAE is not None:
                visual_input = auxiliary_batch.state
                visual_input = torch.cat([t.image for t in visual_input], 0)
                visual_target = auxiliary_batch.next_state
                visual_target = torch.cat([t.image for t in visual_target], 0)
                action_logit = torch.cat(auxiliary_batch.action_logit, 0)
                tae_output = self.model.tAE(visual_input, action_logit)
                tae_loss = torch.sum((tae_output - visual_target).pow(2))

            # visual target classification loss
            # print('vtc pred')
            if (self.model.visual_target_classificator is not None) and (
                    self.vtcFuncInd is not None):
                vtc_inputs = auxiliary_batch.next_state
                vtc_target = torch.from_numpy(
                    np.array(auxiliary_batch.vtcind)
                ).type_as(
                    vtc_inputs[0].image.data
                )

                vtc_out = self.model.visual_target_classificator(vtc_inputs)
                vtc_loss = F.binary_cross_entropy_with_logits(
                    vtc_out, Variable(vtc_target).view(-1, 1))

            # Value function replay
            # Non-skewed sampling from experience buffer
            # print('val pred')
            auxiliary_seq_samples = self.memory.sample_sequence(
                self.args.batch_size)
            auxiliary_seq_batch = ReplayFrame(*zip(*auxiliary_seq_samples))
            nauxsamples = len(auxiliary_seq_samples)
            hidden_states = None
            batch_vrp_ouput = []
            for i in range(nauxsamples):
                _, aux_value, hidden_states = self.model.value_replay_predictor(
                    auxiliary_seq_samples[i].state, hidden_states)
                batch_vrp_ouput.append(aux_value)
            R_vr = 0.0
            if not (auxiliary_seq_samples[-1].terminal):
                _, aux_value, hidden_states = self.model.value_replay_predictor(
                    auxiliary_seq_samples[-1].next_state, hidden_states)
                R_vr = aux_value.detach()
            batch_vrp_target = self.compute_reversed_reward(
                R_vr, auxiliary_seq_batch.reward)
            for i in range(nauxsamples):
                value_replay_loss = value_replay_loss + 0.5 * \
                    (batch_vrp_target[i] - batch_vrp_ouput[i]).pow(2)

            # Reward Prediction loss
            # Skewed-Sampling from experience buffer # TODO
            # print('rew pred')
            if self.model.reward_predictor is not None:
                skewed_samples = self.memory.skewed_sample(
                    self.args.batch_size)
                skewed_batch = ReplayFrame(*zip(*skewed_samples))
                batch_rp_input = []
                batch_rp_output = []

                for i in range(0, len(skewed_samples),
                               self.memory.num_frame_rp + 1):
                    rp_input = skewed_batch.state[
                        i: i + self.memory.num_frame_rp
                    ]
                    rp_output = skewed_batch.reward[
                        i + self.memory.num_frame_rp
                    ]

                    batch_rp_input.append(rp_input)
                    batch_rp_output.append(rp_output)

                rp_predicted = self.model.reward_predictor(batch_rp_input)
                batch_rp_output = torch.from_numpy(
                    np.array(batch_rp_output)).type_as(rp_predicted.data)
                reward_prediction_loss = torch.sum(
                    (rp_predicted - Variable(batch_rp_output)).pow(2))

            # Language Prediction Loss
            # Positive Skewed Sampled from Experience replay
            # print('lang pred')
            if self.model.language_predictor is not None and(
                self.target_keywords is not None and (
                    len(self.target_keywords) > 0)):
                auxiliary_pskew_samples = self.memory.positive_skewed_sample(
                    self.args.batch_size, 0)
                if len(auxiliary_pskew_samples) > 0:
                    auxiliary_pskew_batch = ReplayFrame(
                        *zip(*auxiliary_pskew_samples))

                    state_pskew_input = auxiliary_pskew_batch.state
                    visual_pskew_input = torch.cat(
                        [t.image for t in state_pskew_input], 0)
                    visual_pskew_target = [
                        self.identify_lp_target_keywords(
                            t.mission) for t in state_pskew_input
                    ]

                    lp_out = self.model.language_predictor(visual_pskew_input)
                    lp_out_softmax = F.log_softmax(lp_out, dim=1)

                    for i in range(lp_out_softmax.size(0)):
                        elt_loss = 0
                        for k in range(len(visual_pskew_target[i])):
                            elt_loss = elt_loss + F.nll_loss(
                                lp_out_softmax[i, :].unsqueeze(0),
                                Variable(
                                    torch.Tensor(
                                        [visual_pskew_target[i][k]]
                                    ).type_as(
                                        state_pskew_input[0].mission.data
                                    )
                                ),
                                self.lpweight
                            )
                        elt_loss = elt_loss / max(
                            len(visual_pskew_target[i]), 1)

                        language_prediction_loss += elt_loss

                    language_prediction_loss /= lp_out_softmax.size(0)

        # print('optim')
        self.optimizer.zero_grad()

        # Back-propagation
        total_loss = (policy_loss + self.args.value_loss_coef * value_loss +
                      reward_prediction_loss + tae_loss + vtc_loss +
                      language_prediction_loss + value_replay_loss)

        # print('total_loss: ', total_loss)

        # because of replay memory, we may need to retain graph
        total_loss.backward()  # retain_graph=self.usememory

        # Apply updates
        self.optimizer.step()
        return total_loss.cpu().data.numpy()[0, 0]

    def _train_episodes(self, max_iters, start_epoch, start_step,
                        max_fwd_steps=None, path='.'):

        step = start_step
        step_elapsed = 0
        updatecount = 0
        updateprintcount = 0

        chekoutindex = int(step / self.checkpoint_every) + 1  # 1

        self.model.train()

        current_episode = start_epoch
        current_episode_reward = 0.0
        current_episode_step = 0
        total_loss = 0
        print_loss_total = 0  # Reset every log_interval
        epoch_loss_total = 0  # Reset every epoch

        start = time.time()

        state = self.env.reset()
        state = self.process_state(state)
        self.model.reset_hidden_states()
        if not (self.memory is None):
            self.memory.reset_rp_frame_index()

        prev_terminal_end = False
        print_loss_log = False
        runningReward = 0

        while step < max_iters:

            # fill the memory if necessary
            state = self.fill_memory_replay(state)

            # collect the data for the current episode
            episode_length = 0
            values = []
            log_probs = []
            rewards = []
            entropies = []
            # actions = []
            # states = []
            curr_reward = 0.0
            terminal_end = False
            # kind of truncated backprop through the time
            self.model.detach_hidden_states()

            while True:
                episode_length += 1
                current_episode_step += 1
                step_elapsed += 1

                logit, value = self.model(state)

                # Calculate entropy from action probability distribution
                action_probs = F.softmax(logit, dim=1)
                action_log_prob = F.log_softmax(logit, dim=1)
                entropy = -(action_log_prob * action_probs).sum(1)

                # Take an action from distribution
                m = Categorical(action_probs)
                actionV = m.sample()
                action = actionV.data.cpu().numpy()[0]
                select_action_log_proba = action_log_prob.gather(
                    1, actionV.view(-1, 1))

                # Perform the action on the environment
                nextstate, reward, done, _ = self.env.step(action)
                vtcInd = 0
                if not self.vtcFuncInd is None:
                    img = nextstate if not isinstance(
                        nextstate, dict) else nextstate['image']
                    vtcInd = int(self.vtcFuncInd(self.env, img))
                next_state = self.process_state(nextstate)
                reward = self.process_reward(reward)

                # Cumul reward
                curr_reward += reward
                current_episode_reward += reward

                # Store informations
                # # states.append(state)
                # # actions.append(action)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)
                log_probs.append(select_action_log_proba)

                # Save in Memory if necessary
                if not (self.memory is None):
                    frame = ReplayFrame(
                        state,
                        Variable(logit.data.clone()), next_state, reward,
                        Variable(value.data.clone()), done, vtcInd
                    )
                    self.memory.push(frame)

                if done:
                    terminal_end = True
                    state = self.env.reset()
                    state = self.process_state(state)
                    self.model.reset_hidden_states()
                    if not (self.memory is None):
                        self.memory.reset_rp_frame_index()

                    break
                else:
                    state = next_state
                    if (max_fwd_steps is not None) and (
                            episode_length >= max_fwd_steps):
                        break

            R = torch.zeros(1, 1).type_as(values[-1].data)
            if not terminal_end:
                _, value = self.model(state)
                R = value.data
            values.append(Variable(R))

            if sum(rewards) > 0:
                print('SUCCESS Reward Optim: ', sum(rewards),
                      ' - ', current_episode)

            runningReward = runningReward * 0.99 + sum(rewards) * 0.01

            loss = self.optimize_model(values, log_probs, rewards, entropies)

            # if print_loss_log or sum(rewards) > 0:
            #     print('loss: ', current_episode, ': ', loss,
            #           'rew: ', sum(rewards))

            # Record average loss
            total_loss += loss
            print_loss_total += loss
            epoch_loss_total += loss

            step += episode_length
            updatecount += 1
            updateprintcount += 1
            epoch_loss_avg = epoch_loss_total / updatecount

            # log interval
            if (current_episode % self.log_interval == 0) and (
                    step_elapsed > 0):
                print_loss_avg = print_loss_total / updateprintcount
                print_loss_total = 0
                updateprintcount = 0
                print_loss_log = False
                log_msg = 'Progress: %d %d%%, pr Loss: %.4f, Ep Loss: %.4f' % (
                    step,
                    step / max_iters * 100,
                    print_loss_avg,
                    epoch_loss_avg
                )
                print(log_msg)
                print('Episode {}:{} -> Cur Rew {} Episode Loss {} Run Rew {}'.
                      format(current_episode, current_episode_step,
                             current_episode_reward, epoch_loss_total,
                             runningReward))

            prev_terminal_end = False
            if terminal_end:
                if current_episode_reward > 0:
                    prev_terminal_end = True
                    print_loss_log = True
                current_episode += 1
                epoch_loss_total = 0
                updatecount = 0
                current_episode_reward = 0.0
                current_episode_step = 0
                self.optimizer.update(epoch_loss_avg, current_episode)

            # Checkpoint
            if step >= chekoutindex * self.checkpoint_every or step >= max_iters:
                Checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=current_episode,
                    step=step).save(
                    self.expt_dir)
                chekoutindex += 1

        pass

    def train(
            self, max_iters, max_fwd_steps=None,
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
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(
                    optim.Adam(
                        self.model.parameters(), lr=self.args.lr),
                    max_grad_norm=self.args.max_grad_norm)
            self.optimizer = optimizer

        self._train_episodes(
            max_iters, start_epoch, step, max_fwd_steps, path=path)

        return self.model

    def test(self, max_iters=None, render=False):

        self.model.eval()

        state = self.env.reset()
        if isinstance(state, dict) and 'mission' in state:
            mission = state['mission']
        else:
            mission = ''
        if not (mission == ''):
            print(mission)
        state = self.process_state(state)
        self.model.reset_hidden_states()

        step = 0
        episode_length = 0
        reward_sum = 0

        start_time = time.time()

        while True:
            episode_length += 1
            step += 1

            logit, value = self.model(state)

            # Calculate entropy from action probability distribution
            action_probs = F.softmax(logit, dim=1)
            m = Categorical(action_probs)
            action = m.sample()
            action = action.data.cpu().numpy()[0]

            nextstate, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
                time.sleep(0.1)
            next_state = self.process_state(nextstate)

            if max_iters is not None:
                done = done or step >= max_iters
            reward_sum += reward

            if not done:
                state = next_state
            else:
                print(
                    "Time {}, steps: {}, FPS: {:.0f}, rew: {}, len: {}".format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)
                                      ),
                        step,
                        step / (time.time() - start_time),
                        reward_sum,
                        episode_length
                    ), ' ', mission
                )
                reward_sum = 0
                episode_length = 0
                state = self.env.reset()
                if isinstance(state, dict) and 'mission' in state:
                    mission = state['mission']
                else:
                    mission = ''

                if not (mission == ''):
                    print(mission)
                state = self.process_state(state)
                self.model.reset_hidden_states()
                time.sleep(1)

            if (max_iters is not None) and (step >= max_iters):
                break
