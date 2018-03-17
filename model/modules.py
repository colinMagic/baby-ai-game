import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math
from functools import reduce
import operator
from collections import namedtuple, deque, OrderedDict


class VisionModule(nn.Module):

    def __init__(self, channels=[3, 1], kernels=[8], strides=None):
        '''
            Use the same hyperparameter settings denoted in the paper
        '''

        super(VisionModule, self).__init__()
        assert len(channels) > 1, "The channels length must be greater than 1"
        assert (
            len(channels) - 1 == len(kernels)
        ), "The array lengths must be equals"
        if strides is None:
            strides = [1] * len(kernels)
        assert len(kernels) == len(strides), "The array lengths must be equals"

        self.channels = list(channels)
        self.kernels = list(kernels)
        self.strides = list(strides)

        ordered_modules = OrderedDict()
        for i in range(len(channels) - 1):
            conv = nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernels[i],
                stride=strides[i]
            )
            ordered_modules["conv{}".format(i + 1)] = conv
            # setattr(self, "conv{}".format(i + 1), conv)
        # self.num_layers = len(channels) - 1

        self.conv = nn.Sequential(ordered_modules)

    def forward(self, x):
        # # x is input image with shape [3, 84, 84]
        # out = x
        # for i in range(self.num_layers):
        #    out = getattr(self, "conv{}".format(i + 1))(out)
        # return out
        return self.conv(x)

    def get_output_shape(self, input_shape):
        if len(input_shape) == 1:
            h, w = input_shape[0], input_shape[0]
        elif len(input_shape) == 2:
            h, w = input_shape[0], input_shape[1]
        elif len(input_shape) == 3:
            h, w = input_shape[1], input_shape[2]
        else:
            h, w = input_shape[-2], input_shape[-1]

        for i in range(len(self.channels) - 1):
            kernel_size = self.kernels[i]
            stride = self.strides[i]
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)

            h = math.floor((h - 1 * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            w = math.floor((w - 1 * (kernel_size[1] - 1) - 1) / stride[1] + 1)

            h = int(h)
            w = int(w)

        return [self.channels[-1], h, w]

    def build_deconv(self):
        ordered_modules = OrderedDict()
        for i in range(len(self.channels) - 2, -1, -1):
            deconv = nn.ConvTranspose2d(
                in_channels=self.channels[i + 1],
                out_channels=self.channels[i],
                kernel_size=self.kernels[i],
                stride=self.strides[i]
            )
            ordered_modules["deconv{}".format(
                len(self.channels) - i - 1)] = deconv
        return nn.Sequential(ordered_modules)


class LanguageModule(nn.Module):

    def __init__(self, vocab_size=10, embed_dim=128, hidden_size=128):
        '''
            Use the same hyperparameter settings denoted in the paper
        '''

        super(LanguageModule, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=1,
            batch_first=True)

    def forward(self, x, input_lengths=None):
        embedded_input = self.embeddings(x)
        if input_lengths is not None:  # Variable lenghts
            embedded_input = nn.utils.rnn.pack_padded_sequence(
                embedded_input, input_lengths, batch_first=True)
        out, hn = self.rnn(embedded_input)
        if input_lengths is not None:  # Variable lenghts
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)
        h = hn

        return h

    def sort_data(self, x, input_lengths):
        maxlen = max(input_lengths)
        sorted_x = Variable(
            torch.zeros(
                len(input_lengths),
                maxlen).type_as(
                x[0].data))
        len_tensor = torch.from_numpy(np.array(input_lengths)).type_as(
            x[0].data)
        sorted_len, sorted_index = torch.sort(len_tensor, descending=True)
        inputs_len = []
        for i in range(len(input_lengths)):
            sorted_x[i, 0:sorted_len[i]] = x[sorted_index[i]][
                0:sorted_len[i]
            ].view(-1)
            inputs_len.append(sorted_len[i])

        return sorted_x, inputs_len, sorted_index

    def forward_reordering(self, x, input_lengths):
        if input_lengths is None:
            return self(x)

        if isinstance(input_lengths, torch.autograd.Variable):
            set_inputs = set(input_lengths.view(-1).cpu().data.tolist())
            input_lengths = input_lengths.view(-1).cpu().data.tolist()
        elif isinstance(input_lengths, torch.Tensor):
            set_inputs = set(input_lengths.view(-1).cpu().tolist())
            input_lengths = input_lengths.view(-1).cpu().tolist()
        else:
            set_inputs = set(input_lengths)

        if len(set_inputs) == 1:
            list_set_inputs = list(set_inputs)
            size = list_set_inputs[0]
            return self(x[:, 0:size])
        else:
            sorted_x, inputs_len, sorted_index = self.sort_data(
                x, input_lengths)
            result = self(sorted_x, inputs_len)
            result = torch.index_select(
                result, dim=1, index=Variable(sorted_index)
            )
            return result

    def LP_Inv_Emb(self, x):
        return F.linear(x, self.embeddings.weight)

# Conditional Batch norm is a classical Batch Norm Module
# with the affine parameter set to False


class ConditionalBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ConditionalBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(
                'expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, input, gamma, beta):
        return F.batch_norm(
            input, self.running_mean, self.running_var, gamma, beta,
            self.training, self.momentum, self.eps)

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, momentum={momentum}'.format(
                name=self.__class__.__name__, **self.__dict__)
        )


class MixingModule(nn.Module):

    def __init__(self):
        super(MixingModule, self).__init__()

    def forward(self, visual_encoded, instruction_encoded=None):
        '''
            Argument:
                visual_encoded: output of vision module, shape [batch_size, 64, 7, 7]
                instruction_encoded: hidden state of language module, shape [batch_size, 1, 128]
        '''
        batch_size = visual_encoded.size()[0]
        visual_flatten = visual_encoded.view(batch_size, -1)
        if instruction_encoded is not None:
            instruction_flatten = instruction_encoded.view(batch_size, -1)
            mixed = torch.cat([visual_flatten, instruction_flatten], dim=1)
            return mixed
        else:
            return visual_flatten


class ActionModule(nn.Module):

    def __init__(self, input_size=3264, hidden_size=256):
        super(ActionModule, self).__init__()
        self.hidden_size = hidden_size

        self.lstm_1 = nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size)
        self.lstm_2 = nn.LSTMCell(
            input_size=hidden_size, hidden_size=hidden_size)

    def repackage_hidden(self, h):
        """ Wraps hidden states in new Variables,
            to detach them from their history.
        """
        if isinstance(h, torch.autograd.Variable):
            return torch.autograd.Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def hidden_state_shape(self):
        return (2, 2, self.hidden_size)

    def forward(self, x, hidden_states=None):
        '''
            Argument:
                x: x is output from the Mixing Module, as shape [batch_size, 1, 3264]
        '''
        # Feed forward
        assert x.dim() == 2, 'the dimension of x should be 2'

        hidden_size = self.hidden_size

        if hidden_states is None:
            hidden_states_1 = (
                Variable(torch.zeros(x.size(0), hidden_size).type_as(x.data)),
                Variable(torch.zeros(x.size(0), hidden_size).type_as(x.data))
            )
            hidden_states_2 = (
                Variable(torch.zeros(x.size(0), hidden_size).type_as(x.data)),
                Variable(torch.zeros(x.size(0), hidden_size).type_as(x.data))
            )
        else:
            # batch_size, 2, 2, self.hidden_size
            hidden_states_1 = hidden_states[:, 0]
            hidden_states_2 = hidden_states[:, 1]

            hidden_states_1 = (hidden_states_1[:, 0], hidden_states_1[:, 1])
            hidden_states_2 = (hidden_states_2[:, 0], hidden_states_2[:, 1])

        h1, c1 = self.lstm_1(x, hidden_states_1)

        h2, c2 = self.lstm_2(h1, hidden_states_2)

        x2 = h2

        # Update current hidden state
        hidden_states_1 = torch.cat([h1.unsqueeze(1), c1.unsqueeze(1)], dim=1)
        hidden_states_2 = torch.cat([h2.unsqueeze(1), c2.unsqueeze(1)], dim=1)

        states = torch.cat(
            [hidden_states_1.unsqueeze(1), hidden_states_2.unsqueeze(1)],
            dim=1
        )

        # Return the hidden state of the upper layer
        return x2, states

# We should do a ConditionalBatchNormModule, ClassifierModule, HistoricalRNN Module
# RL Module


SavedAction = namedtuple('SavedAction', ['action', 'value'])


class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)

        probs = F.softmax(x, dim=1)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1, keepdim=True)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=1)
        probs = F.softmax(x, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy


class AddBias(nn.Module):

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = Variable(torch.zeros(action_mean.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        if deterministic is False:
            noise = Variable(torch.randn(action_std.size()))
            if action_std.is_cuda:
                noise = noise.cuda()
            action = action_mean + action_std * noise
        else:
            action = action_mean
        return action

    def logprobs_and_entropy(self, x, actions):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(
            2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(-1, keepdim=True)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return action_log_probs, dist_entropy


class Policy(nn.Module):

    def __init__(self, action_space, input_size=256, hidden_size=128):
        super(Policy, self).__init__()
        self.action_space = action_space

        self.affine1 = nn.Linear(input_size, hidden_size)
        self.action_head = Categorical(hidden_size, action_space)
        self.value_head = nn.Linear(hidden_size, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return action_scores, state_values

    def forward_without_actions(self, x):
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)

        return x, state_values

    def sample(self, x, deterministic):
        return self.action_head.sample(x, deterministic)

    def logprobs_and_entropy(self, x, actions):
        return self.action_head.logprobs_and_entropy(x, actions)

    def tAE(self, action_logits):
        '''
        Temporal Autoencoder sub-task
        Argument:
            action_logits: shape [1, action_space]

        Return:
            output has shape: [1, hidden_size] # [1, 128]
                which is the inverse transformation of the Linear
                module corresponding to the policy (action_head)
        '''
        bias = torch.unsqueeze(
            self.action_head.linear.bias, 0).repeat(
            action_logits.size()[0], 1)

        output = action_logits - bias
        output = F.linear(
            output, torch.transpose(
                self.action_head.linear.weight, 0, 1))

        return output


class TemporalAutoEncoder(nn.Module):

    def __init__(self, policy_network, vision_module,
                 input_size=128, vision_encoded_shape=[64, 7, 7]):
        super(TemporalAutoEncoder, self).__init__()

        self.policy_network = policy_network
        self.vision_module = vision_module
        self.vision_encoded_shape = vision_encoded_shape
        self.input_size = input_size
        self.hidden_size = reduce(operator.mul, vision_encoded_shape, 1)

        self.linear_1 = nn.Linear(input_size, self.hidden_size)
        self.deconv = self.vision_module.build_deconv()

    def forward(self, visual_input, logit_action, deconvFlag=True):
        '''
        Argument:
            visual_encoded: output from the visual module, has shape [1, 64, 7, 7]
            logit_action: output action logit from policy, has shape [1, 10]
        '''
        visual_encoded = self.vision_module(visual_input)

        action_out = self.policy_network.tAE(logit_action)  # [1, 128]
        action_out = self.linear_1(action_out)
        action_out = action_out.view(
            action_out.size()[0], *self.vision_encoded_shape)

        out = torch.mul(action_out, visual_encoded)

        if deconvFlag:
            out = self.deconv(out)
        return out


class ICM(nn.Module):

    def __init__(self, policy_network, action_space,
                 input_size=128, hidden_size=3136, act_hid_size=128):
        super(ICM, self).__init__()

        self.policy_network = policy_network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_space = action_space
        self.act_hid_size = act_hid_size
        self.action_space = action_space

        self.linear_1 = nn.Linear(input_size, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.linear_3 = nn.Linear(2 * self.hidden_size, self.act_hid_size)
        self.linear_4 = nn.Linear(self.act_hid_size, self.action_space)

    def forward(self, state, next_state, logit_action):
        '''
        Argument:
            visual_encoded: output from the visual module, has shape [1, 64, 7, 7]
            logit_action: output action logit from policy, has shape [1, 10]
        '''

        action_out = self.policy_network.tAE(logit_action)  # [1, 128]
        action_out = self.linear_1(action_out)
        action_out = action_out.view_as(state)

        out = torch.mul(action_out, state)

        out = self.linear_2(out.view(out.size(0), -1)).view_as(state)

        concatState = torch.cat(
            [
                state.view(state.size(0), -1),
                next_state.view(next_state.size(0), -1)
            ], dim=1
        )
        act_pred = self.linear_3(concatState)
        act_pred = self.linear_4(act_pred)

        return out, act_pred


class RNNStatePredictor(nn.Module):

    def __init__(self, policy_network, vision_module,
                 input_size=128, vision_encoded_shape=[64, 7, 7],
                 ouput_size=1024):
        super(TemporalAutoEncoder, self).__init__()

        self.policy_network = policy_network
        self.vision_module = vision_module
        self.vision_encoded_shape = vision_encoded_shape
        self.input_size = input_size
        self.hidden_size = reduce(operator.mul, vision_encoded_shape, 1)
        self.ouput_size = ouput_size

        self.linear_1 = nn.Linear(input_size, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size, self.ouput_size)

    def forward(self, visual_input, logit_action):
        '''
        Argument:
            visual_encoded: output from the visual module, has shape [1, 64, 7, 7]
            logit_action: output action logit from policy, has shape [1, 10]
        '''
        visual_encoded = self.vision_module(visual_input)

        action_out = self.policy_network.tAE(logit_action)  # [1, 128]
        action_out = self.linear_1(action_out)
        action_out = action_out.view(
            action_out.size()[0], *self.vision_encoded_shape)

        combine = torch.mul(action_out, visual_encoded)

        out = self.linear_2(combine)
        return out


class LanguagePrediction(nn.Module):

    def __init__(self, language_module, vision_module,
                 vision_encoded_shape=[64, 7, 7],
                 hidden_size=128):
        super(LanguagePrediction, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module

        input_size = reduce(operator.mul, vision_encoded_shape, 1)

        self.vision_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU())

    def forward(self, visual_input):

        vision_encoded = self.vision_module(visual_input)

        vision_encoded_flatten = vision_encoded.view(
            vision_encoded.size()[0], -1)
        vision_out = self.vision_transform(vision_encoded_flatten)

        language_predict = self.language_module.LP_Inv_Emb(vision_out)

        return language_predict


class RewardPrediction(nn.Module):

    def __init__(self, vision_module, language_module, mixing_module,
                 num_elts=3, vision_encoded_shape=[64, 7, 7],
                 language_encoded_size=128):
        super(RewardPrediction, self).__init__()

        self.vision_module = vision_module
        self.language_module = language_module
        self.mixing_module = mixing_module
        self.linear = nn.Linear(
            num_elts * (
                reduce(
                    operator.mul,
                    vision_encoded_shape, 1) + language_encoded_size
            ),
            1
        )

    def forward(self, x):
        '''
            x: state including image and instruction,
                    each batch contains 3 (num_elts) images in sequence
                    with the instruction to be encoded
        '''
        batch_visual = []
        batch_instruction = []
        batch_instruction_len = []

        for batch in x:
            visual = [b.image for b in batch]
            instruction = [b.mission for b in batch]
            instruction_len = [b.mission.numel() for b in batch]

            batch_visual.append(torch.cat(visual, 0))
            # batch_instruction.append(torch.cat(instruction, 0))
            batch_instruction.extend(instruction)
            batch_instruction_len.extend(instruction_len)

        if not (self.language_module is None):
            if len(set(batch_instruction_len)) == 1:
                batch_instruction = torch.cat(batch_instruction, 0)
                inputs_len = None
            else:
                maxlen = max(batch_instruction_len)
                sorted_instruction = Variable(torch.zeros(
                    len(batch_instruction_len), maxlen
                ).type_as(
                    batch_instruction[0].data
                ))
                len_tensor = torch.from_numpy(
                    np.array(batch_instruction_len)
                ).type_as(batch_instruction[0].data)
                sorted_len, sorted_index = torch.sort(
                    len_tensor, descending=True)
                inputs_len = []
                for i in range(len(batch_instruction_len)):
                    sorted_instruction[i, 0:sorted_len[i]] = batch_instruction[
                        sorted_index[i]].view(-1)
                    inputs_len.append(sorted_len[i])
                batch_instruction = sorted_instruction

        batch_visual_encoded = self.vision_module(torch.cat(batch_visual, 0))
        batch_instruction_encoded = None
        if not (self.language_module is None):
            if inputs_len is None:
                batch_instruction_encoded = self.language_module(
                    batch_instruction)
            else:
                batch_instruction_encoded = self.language_module(
                    batch_instruction, inputs_len)
                batch_instruction_encoded = torch.index_select(
                    batch_instruction_encoded,
                    dim=1,
                    index=Variable(sorted_index)
                )

        batch_mixed = self.mixing_module(
            batch_visual_encoded, batch_instruction_encoded)
        batch_mixed = batch_mixed.view(len(batch_visual), -1)

        out = self.linear(batch_mixed)
        return out


class VisualTargetClassification(nn.Module):

    def __init__(self, vision_module, language_module, mixing_module,
                 vision_encoded_shape=[64, 7, 7],
                 language_encoded_size=128):
        super(VisualTargetClassification, self).__init__()

        self.vision_module = vision_module
        self.language_module = language_module
        self.mixing_module = mixing_module
        self.linear = nn.Linear(
            reduce(
                operator.mul,
                vision_encoded_shape, 1
            ) + language_encoded_size,
            1
        )

    def forward(self, x):
        '''
            x: states including image and instruction,
        '''
        batch_visual = [b.image for b in x]
        batch_instruction = [b.mission for b in x]
        batch_instruction_len = [b.mission.numel() for b in x]

        if not (self.language_module is None):
            if len(set(batch_instruction_len)) == 1:
                batch_instruction = torch.cat(batch_instruction, 0)
                inputs_len = None
            else:
                maxlen = max(batch_instruction_len)
                sorted_instruction = Variable(torch.zeros(
                    len(batch_instruction_len), maxlen
                ).type_as(
                    batch_instruction[0].data
                ))
                len_tensor = torch.from_numpy(
                    np.array(batch_instruction_len)
                ).type_as(batch_instruction[0].data)
                sorted_len, sorted_index = torch.sort(
                    len_tensor, descending=True)
                inputs_len = []
                for i in range(len(batch_instruction_len)):
                    sorted_instruction[i, 0:sorted_len[i]] = batch_instruction[
                        sorted_index[i]].view(-1)
                    inputs_len.append(sorted_len[i])
                batch_instruction = sorted_instruction

        batch_visual_encoded = self.vision_module(torch.cat(batch_visual, 0))
        batch_instruction_encoded = None
        if not (self.language_module is None):
            if inputs_len is None:
                batch_instruction_encoded = self.language_module(
                    batch_instruction)
            else:
                batch_instruction_encoded = self.language_module(
                    batch_instruction, inputs_len)
                batch_instruction_encoded = torch.index_select(
                    batch_instruction_encoded,
                    dim=1,
                    index=Variable(sorted_index)
                )

        batch_mixed = self.mixing_module(
            batch_visual_encoded, batch_instruction_encoded)
        batch_mixed = batch_mixed.view(len(batch_visual), -1)

        out = self.linear(batch_mixed)

        # Use F.binary_cross_entropy_with_logits() for computing the
        # classification loss on this output
        return out
