#!/usr/bin/env python3

import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import operator
from functools import reduce
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from model import RLAgent, Checkpoint
from model import get_args
from model import PolicyModel

import gym
from gym import spaces
import gym_minigrid
import gym_minigrid.minigrid as minigrid

args = get_args()

if args.num_steps == 0:
    args.num_steps = None

args.checkpoint_every = int(args.max_iters / (args.num_checkpoints + 1)) + 1

assert args.algo in ['a2c', 'ppo']


def make_env_fn(env_id, seed, rank=0):
    def _create():
        env = gym.make(env_id)
        if hasattr(env, 'seed'):
            env.seed(seed + rank)
        return env

    return _create


def main():

    os.environ['OMP_NUM_THREADS'] = '1'

    max_words = 100

    # Build the vocabulary dictionnary of the agent
    vocab_words = []
    specials = ['<pad>', '<eos>', '<unk>']

    vocab_dict = {}
    vocab_dict.update({tok: i for i, tok in enumerate(specials)})
    vocab_dict.update({tok: i + len(specials)
                       for i, tok in enumerate(vocab_words)})

    def processString(text):
        if (text is None) or (text == ''):
            return [vocab_dict['<eos>']]
        else:
            words = text.strip().lower().split()
            word_ids = []
            for i in range(len(words)):
                key = words[i].strip()
                if key == '':
                    continue
                elif key in vocab_dict:
                    word_ids.append(vocab_dict[key])
                elif len(vocab_words) + len(specials) < max_words:
                    vocab_dict[key] = len(vocab_words) + len(specials)
                    vocab_words.append(key)
                    word_ids.append(vocab_dict[key])
                else:
                    word_ids.append(vocab_dict['<unk>'])
            word_ids.append(vocab_dict['<eos>'])
            return word_ids

    def getCurrentLanguageVocab(text):
        return vocab_words

    # Create the model arguments (No need in test mode when loading a saved
    # model)
    if args.load_checkpoint is not None:
        model = None
        hidden_state_shape = None
    else:
        vocab_size = max_words
        env = make_env_fn(args.env_name, 0, 0)()
        model_args = {
            'img_shape': [3, 7, 7],
            'action_space': env.action_space,
            'channels': [3, 16, 32],
            'kernels': [4, 3],
            'strides': None,
            'langmod': True,  # False, #
            'vocab_size': vocab_size,
            'embed_dim': 64,
            'langmod_hidden_size': 64,
            'actmod_hidden_size': 128,
            'policy_hidden_size': 64,
        }
        model = PolicyModel(**model_args)
        hidden_state_shape = model.hidden_state_shape()

    # Create the RL agent based on the environment
    if isinstance(env.observation_space, spaces.Dict):
        observation_space = env.observation_space.__dict__['spaces']['image']
    else:
        observation_space = env.observation_space

    agent = RLAgent(
        make_env_fn, args, device=None,
        expt_dir=args.expt_dir,
        checkpoint_every=args.checkpoint_every,
        log_interval=args.log_interval,
        input_scale=observation_space.high[0][0][0],
        reward_scale=1000,
        hidden_state_shape=hidden_state_shape,
        pad_sym='<pad>',
        process_string_func=processString,
        language_vocab_func=getCurrentLanguageVocab,
        model=model
    )

    textdata_maxlen = 30

    if args.load_checkpoint is not None:
        print(
            "loading checkpoint from {}".format(
                os.path.join(
                    args.expt_dir,
                    Checkpoint.CHECKPOINT_DIR_NAME,
                    args.load_checkpoint)
            )
        )
        checkpoint_path = os.path.join(
            args.expt_dir,
            Checkpoint.CHECKPOINT_DIR_NAME,
            args.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)

        model = checkpoint.model
        vocab = checkpoint.vocab

        agent.model = model

        vocab_words = vocab
        vocab_dict = {}
        vocab_dict.update({tok: i for i, tok in enumerate(specials)})
        vocab_dict.update({tok: i + len(specials)
                           for i, tok in enumerate(vocab_words)})

        print("Finishing Loading")

        # test the agent
        agent.test(
            max_iters=args.max_iters,
            textdata_maxlen=textdata_maxlen,
            render=args.render
        )

    else:

        # train the agent
        agent.train(
            max_iters=args.max_iters,
            textdata_maxlen=textdata_maxlen,
            num_steps=args.num_steps,
            resume=args.resume,
            optimizer=None,
            path='.'
        )


if __name__ == '__main__':
    main()