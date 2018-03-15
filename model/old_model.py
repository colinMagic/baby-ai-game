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


class Model(nn.Module):

    def __init__(self, img_shape, action_space,
                 channels=[3, 1], kernels=[8], strides=None, langmod=False,
                 vocab_size=10, embed_dim=128, langmod_hidden_size=128,
                 actmod_hidden_size=256,
                 policy_hidden_size=128,
                 tae=False,
                 langmod_pred=False,
                 visual_classifier=False,
                 rew_pred=False,
                 num_frames_reward_prediction=3,
                 **kwargs):
        super(Model, self).__init__()

        # Core modules
        self.vision_m = VisionModule(channels, kernels, strides)
        vision_encoded_shape = self.vision_m.get_output_shape(img_shape)
        vision_encoded_dim = reduce(operator.mul, vision_encoded_shape, 1)
        self.language_m = None
        if langmod:
            self.language_m = LanguageModule(
                vocab_size, embed_dim, langmod_hidden_size)
        else:
            langmod_hidden_size = 0
        self.mixing_m = MixingModule()
        self.action_m = ActionModule(
            input_size=vision_encoded_dim + langmod_hidden_size,
            hidden_size=actmod_hidden_size)

        # Action selection and Value Critic
        self.policy = Policy(
            action_space=action_space,
            input_size=actmod_hidden_size, hidden_size=policy_hidden_size)

        # Auxiliary networks
        self.tAE = None
        if tae:
            self.tAE = TemporalAutoEncoder(
                self.policy, self.vision_m,
                input_size=policy_hidden_size,
                vision_encoded_shape=vision_encoded_shape
            )
        self.language_predictor = None
        if langmod and langmod_pred:
            self.language_predictor = LanguagePrediction(
                self.language_m, self.vision_m,
                vision_encoded_shape=vision_encoded_shape,
                hidden_size=langmod_hidden_size
            )
        self.reward_predictor = None
        if rew_pred:
            self.reward_predictor = RewardPrediction(
                self.vision_m, self.language_m, self.mixing_m,
                num_elts=num_frames_reward_prediction,
                vision_encoded_shape=vision_encoded_shape,
                language_encoded_size=langmod_hidden_size
            )
        self.visual_target_classificator = None
        if visual_classifier:
            self.visual_target_classificator = VisualTargetClassification(
                self.vision_m, self.language_m, self.mixing_m,
                vision_encoded_shape=vision_encoded_shape,
                language_encoded_size=langmod_hidden_size
            )

    def reset_hidden_states(self):
        self.action_m.reset_hidden_states()

    def detach_hidden_states(self):
        self.action_m.detach_hidden_states()

    def mask_hidden_states(self, h, mask):
        if h is None:
            return None
        if mask is None:
            return h
        if isinstance(h, torch.autograd.Variable):
            return h * mask
        else:
            return tuple(self.mask_hidden_states(v, mask) for v in h)

    def size_hidden_states(self, h, dim=0):
        if isinstance(h, torch.autograd.Variable):
            return h.size(dim)
        else:
            return self.size_hidden_states(h[0], dim)

    def compute_input_lenght_from_padding(self, x, pad_sym=0):
        mask = x.data.eq(pad_sym)
        x_len = x.size(1) - torch.sum(mask, dim=1)
        x_len = x_len.int()
        return x_len.tolist()

    def forward(self, x):
        '''
        Argument:

            img: environment image, shape [batch_size, 84, 84, 3]
            instruction: natural language instruction [batch_size, seq]
        '''

        vision_out = self.vision_m(x.image)
        language_out = None
        if not (self.language_m is None):
            mission_len = self.compute_input_lenght_from_padding(x.mission)
            language_out = self.language_m.forward_reordering(
                x.mission, mission_len)
        mix_out = self.mixing_m(vision_out, language_out)
        action_out = self.action_m(mix_out)

        action_prob, value = self.policy(action_out)

        return action_prob, value

    def forward_with_hidden_states(self, x, hidden_states=None, mask=None):
        '''
        Argument:

            img: environment image, shape [batch_size, 84, 84, 3]
            instruction: natural language instruction [batch_size, seq]
        '''

        vision_out = self.vision_m(x.image)
        language_out = None
        if not (self.language_m is None):
            mission_len = self.compute_input_lenght_from_padding(x.mission)
            language_out = self.language_m.forward_reordering(
                x.mission, mission_len)

        mix_out = self.mixing_m(vision_out, language_out)

        if hidden_states is None:
            action_out, hidden_states = self.action_m.forward_with_hidden_states(
                mix_out, hidden_states)
        else:
            if mix_out.size(0) == self.size_hidden_states(hidden_states, 1):
                if mask is not None:
                    hidden_states = self.mask_hidden_states(
                        hidden_states, mask)
                action_out, hidden_states = self.action_m.forward_with_hidden_states(
                    mix_out, hidden_states)
            else:
                mix_out = mix_out.view(
                    -1,
                    self.size_hidden_states(hidden_states, 1),
                    mix_out.size(1))
                if mask is not None:
                    mask = mask.view(
                        -1, self.size_hidden_states(hidden_states, 1), 1)
                outputs = []
                for i in range(mix_out.size(0)):
                    if mask is not None:
                        hidden_states = self.mask_hidden_states(
                            hidden_states, mask[i])
                    act_out_i, hidden_states = self.action_m.forward_with_hidden_states(
                        mix_out[i], hidden_states)
                    outputs.append(act_out_i)

                action_out = torch.cat(outputs, 0)

        action_prob, value = self.policy(action_out)

        return action_prob, value, hidden_states

    def value_replay_predictor(self, x, hidden_states=None):
        '''
        Argument:

            img: environment image, shape [batch_size, 84, 84, 3]
            instruction: natural language instruction [batch_size, seq]
        '''

        vision_out = self.vision_m(x.image)
        language_out = None
        if not (self.language_m is None):
            language_out = self.language_m(x.mission)
        mix_out = self.mixing_m(vision_out, language_out)
        action_out, hidden_states = self.action_m.forward_with_hidden_states(
            mix_out, hidden_states)

        action_prob, value = self.policy(action_out)

        return action_prob, value, hidden_states
