from __future__ import print_function
import numpy as np
from collections import namedtuple, deque
import os
import time
import shutil
import torch


State = namedtuple('State', ('image', 'mission', 'missionlen'))


def apply_flatten_parameters(m):
    if hasattr(m, 'flatten_parameters'):
        if callable(m.flatten_parameters):
            m.flatten_parameters()


class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model during
    training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using
    sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following
    args; then call that object's save() method to write parameters to disk.

    Args:
        model: model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch
        step (int): number of examples seen within the current epoch

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
    """

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(
            self, model, optimizer, epoch, step, vocab=[], path=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self.vocab = vocab
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir):
        """
        Saves the current model and related training parameters into
        a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in
        Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(
            experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer,
                    'vocab': self.vocab
                    },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from
            those stored on disk
        """
        print("Loading checkpoints from {}".format(path))
        try:
            resume_checkpoint = torch.load(
                os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        except:
            resume_checkpoint = torch.load(
                os.path.join(path, cls.TRAINER_STATE_NAME),
                map_location=lambda storage, loc: storage)
            model = torch.load(
                os.path.join(path, cls.MODEL_NAME),
                map_location=lambda storage, loc: storage)
        # make RNN parameters contiguous
        model = model.apply(apply_flatten_parameters)
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          vocab=resume_checkpoint['vocab'],
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to
        the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made
        (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(
            experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])
