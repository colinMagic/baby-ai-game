import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from functools import reduce
import operator
from .modules import *
from .utils import State


class PolicyInterface(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def hasRecurrence(self):
        raise NotImplementedError

    def hidden_state_shape(self):
        """
        Size of the recurrent state of the model (propagated between steps)
        """
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(
            x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(
            x, actions)
        return value, action_log_probs, dist_entropy, states


class PolicyModel(PolicyInterface):

    def __init__(self, img_shape, action_space,
                 channels=[3, 1], kernels=[8], strides=None, langmod=False,
                 vocab_size=10, embed_dim=128, langmod_hidden_size=128,
                 actmod_hidden_size=256,
                 policy_hidden_size=128,
                 **kwargs):
        super(PolicyModel, self).__init__()

        assert action_space.__class__.__name__ == "Discrete"
        action_space = action_space.n

        # Vision module
        self.vision = VisionModule(channels, kernels, strides)
        vision_encoded_shape = self.vision.get_output_shape(img_shape)
        vision_encoded_dim = reduce(operator.mul, vision_encoded_shape, 1)

        # Language module
        self.language = None
        if langmod:
            self.language = LanguageModule(
                vocab_size, embed_dim, langmod_hidden_size)
        else:
            langmod_hidden_size = 0

        # Mixing module
        self.mixing = MixingModule()

        # Action module
        self.action = ActionModule(
            input_size=vision_encoded_dim + langmod_hidden_size,
            hidden_size=actmod_hidden_size)

        # Action selection and Value Critic
        self.policy = Policy(
            action_space=action_space,
            input_size=actmod_hidden_size, hidden_size=policy_hidden_size)

        self.dist = self.policy
        self.policyWithoutDirectActionComputation = True

    def hasRecurrence(self):
        return True

    def hidden_state_shape(self):
        return self.action.hidden_state_shape()

    def forward(self, x, hidden_states=None, mask=None):
        '''
        Argument:

        x:
            image: environment image, shape [batch_size, 84, 84, 3]
            mission: natural language instruction [batch_size, seq]
            missionlen: len of each instruction [batch_size, 1]

        hidden_states: hidden state of the network
        mask: mask to be used

        '''

        vision_out = self.vision(x.image)
        language_out = None
        if not (self.language is None):
            language_out = self.language.forward_reordering(
                x.mission, x.missionlen)

        mix_out = self.mixing(vision_out, language_out)

        if hidden_states is None:
            action_out, hidden_states = self.action(mix_out, hidden_states)
        else:
            if mix_out.size(0) == hidden_states.size(0):
                if mask is not None:
                    shape = hidden_states.size()
                    hidt = hidden_states.view(hidden_states.size(0), -1) * mask
                    hidden_states = hidt.view(shape)
                action_out, hidden_states = self.action(mix_out, hidden_states)
            else:
                mix_out = mix_out.view(
                    -1,
                    hidden_states.size(0),
                    mix_out.size(1)
                )
                if mask is not None:
                    mask = mask.view(-1, hidden_states.size(0), 1)
                outputs = []
                for i in range(mix_out.size(0)):
                    if mask is not None:
                        shape = hidden_states.size()
                        hidt = hidden_states.view(
                            hidden_states.size(0), -1
                        ) * mask[i]
                        hidden_states = hidt.view(shape)
                    act_out_i, hidden_states = self.action(
                        mix_out[i],
                        hidden_states
                    )
                    outputs.append(act_out_i)

                action_out = torch.cat(outputs, 0)

        if self.policyWithoutDirectActionComputation:
            actionProb, value = self.policy.forward_without_actions(action_out)
        else:
            actionProb, value = self.policy(action_out)

        return value, actionProb, hidden_states
