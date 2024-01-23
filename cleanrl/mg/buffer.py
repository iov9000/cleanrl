# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

from absl import logging
import numpy as np


# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.
ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

# A prefix that can not collide with variable names for checkpoint files.
STORE_FILENAME_PREFIX = '$store$_'


def modulo_range(start, length, modulo):
    for i in range(length):
        yield (start + i) % modulo


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
    """Returns a array with the indices of cursor-related invalid transitions.

    There are update_horizon + stack_size invalid indices:
      - The update_horizon indices before the cursor, because we do not have a
        valid N-step transition (including the next state).
      - The stack_size indices on or immediately after the cursor.
    If N = update_horizon, K = stack_size, and the cursor is at c, invalid
    indices are:
      c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.

    It handles special cases in a circular buffer in the beginning and the end.

    Args:
      cursor: int, the position of the cursor.
      replay_capacity: int, the size of the replay memory.
      stack_size: int, the size of the stacks returned by the replay memory.
      update_horizon: int, the agent's update horizon.
    Returns:
      np.array of size stack_size with the invalid indices.
    """
    assert cursor < replay_capacity
    return np.array(
        [(cursor - update_horizon + i) % replay_capacity
         for i in range(stack_size + update_horizon)])


class MultiStepReplayBuffer(object):
    """A simple out-of-graph Replay Buffer.

    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.

    When the states consist of stacks of observations storing the states is
    inefficient. This class writes observations and constructs the stacked states
    at sample time.

    Attributes:
      add_count: int, counter of how many transitions have been added (including
        the blank ones at the beginning of an episode).
      invalid_range: np.array, an array with the indices of cursor-related invalid
        transitions
      episode_end_indices: set[int], a set of indices corresponding to the
        end of an episode.
    """

    def __init__(self, opt,
                 observation_shape,
                 stack_size,
                 replay_capacity,
                 batch_size,
                 update_horizon=1,
                 nstep_distribution="beta",  # "uniform", "beta" or "fix"
                 mode='last',
                 gamma=0.99,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32,
                 checkpoint_duration=4,
                 keep_every=None):
        """Initializes OutOfGraphReplayBuffer.

        Args:
          observation_shape: tuple of ints.
          stack_size: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          batch_size: int.
          update_horizon: int, length of update ('n' in n-step update).
            nstep_distribution: str, the distribution of n in n-step update.
            "uniform" means a uniform distribution between 1 and max_nstep.
            "beta" means a beta distribution between 1 and max_nstep where the
                parameter a is 10 and the parameter b is 1 and being
                annealed so that it starts at 0.8 and ends at 10.
            "fix" means a fix value of n.
          mode: bool, whether to use multi-grid n-step update.
          gamma: float, the discount factor.
          max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
          extra_storage_types: list of ReplayElements defining the type of the extra
            contents that will be stored and returned by sample_transition_batch.
          observation_dtype: np.dtype, type of the observations. Defaults to
            np.uint8 for Atari 2600.
          terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
            Atari 2600.
          action_shape: tuple of ints, the shape for the action vector. Empty tuple
            means the action is a scalar.
          action_dtype: np.dtype, type of elements in the action.
          reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
            means the reward is a scalar.
          reward_dtype: np.dtype, type of elements in the reward.
          checkpoint_duration: int, how many iterations a checkpoint is kept for.
          keep_every: Optional (int or None), keep all checkpoints == 0 % this
            number. Set to None to disable.

        Raises:
          ValueError: If replay_capacity is too small to hold at least one
            transition.
        """
        assert isinstance(observation_shape, tuple)
        if replay_capacity < update_horizon + stack_size:
            raise ValueError('There is not enough capacity to cover '
                             'update_horizon and stack_size.')

        logging.info(
            'Creating a %s replay memory with the following parameters:',
            self.__class__.__name__)
        logging.info('\t observation_shape: %s', str(observation_shape))
        logging.info('\t observation_dtype: %s', str(observation_dtype))
        logging.info('\t terminal_dtype: %s', str(terminal_dtype))
        logging.info('\t stack_size: %d', stack_size)
        logging.info('\t replay_capacity: %d', replay_capacity)
        logging.info('\t batch_size: %d', batch_size)
        logging.info('\t update_horizon: %d', update_horizon)
        logging.info('\t gamma: %f', gamma)
        logging.info('\t checkpoint_duration: %d', checkpoint_duration)
        logging.info('\t keep_every: %s', str(keep_every))

        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._stack_size = stack_size
        self._state_shape = self._observation_shape + (self._stack_size,)
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._max_nstep = update_horizon
        self._nstep_distribution = nstep_distribution
        self._mode = mode
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._max_sample_attempts = max_sample_attempts
        if extra_storage_types:
            self._extra_storage_types = extra_storage_types
        else:
            self._extra_storage_types = []
        self._create_storage()
        self.add_count = np.array(0)
        self.invalid_range = np.zeros((self._stack_size))
        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon)],
            dtype=np.float32)
        self._next_experience_is_episode_start = True
        self.episode_end_indices = set()
        self._checkpoint_duration = checkpoint_duration
        self._keep_every = keep_every

        self._a, self._b = 10, 1
        self._total_steps = 0

        # UCB optimism in face of uncertainty to choose the best n-step
        self.ucb_mean = np.zeros(self._max_nstep)
        self.ucb_std = np.zeros(self._max_nstep)
        self.ucb_min = np.zeros(self._max_nstep)
        self.ucb_max = np.zeros(self._max_nstep)
        self.ucb_pref = np.zeros(self._max_nstep)
        self.ucb_prob_cache = np.zeros(self._max_nstep)
        self.ucb_alpha = opt.ucb_alpha

        self._diff = np.zeros(self._max_nstep)

        self.ucb_nstep_count = np.zeros(self._max_nstep)
        self.ucb_second_nstep_count = np.zeros(self._max_nstep)
        self.ucb = np.zeros(self._max_nstep)
        self._ucb_total_steps = 0
        self._ucb_lambda = opt.ucb_lambda

    def update_nstep(self, new_nstep):
        self._update_horizon = new_nstep

    def anneal_beta_distribution(self):
        start_value = 0.8
        end_value = 10
        num_time_steps = 1000000
        increment = (end_value - start_value) / num_time_steps
        self._b = start_value + self._total_steps * increment

    def _ucb_prob(self, nstep):
        return np.exp(self.ucb_pref[nstep]) / np.sum(np.exp(self.ucb_pref)+1e-6)

    def update_mean_std(self, mean_q, var_q, min_q, max_q, nstep):
        nstep = nstep - 1

        ucb_pref = self.ucb_pref.copy()
        diff = -(var_q - self.ucb_std[nstep])
        diff = np.clip(diff, -1, 1)
        ucb_pref[nstep] = self.ucb_pref[nstep] + self._ucb_lambda * diff * (1 - self._ucb_prob(nstep))
        for i in range(self._max_nstep):
            if i != nstep:
                ucb_pref[i] = self.ucb_pref[i] - self._ucb_lambda * diff * self._ucb_prob(i)
        self.ucb_pref = ucb_pref

        # TODO: mean_q here is an estimate of the q, not the reward.
        #  shouldn't we use the reward instead?
        self.ucb_mean[nstep] += ((mean_q - self.ucb_mean[nstep]) *
                                 self.ucb_alpha)
        self.ucb_std[nstep] += ((var_q - self.ucb_std[nstep]) * self.ucb_alpha)
        self.ucb_min[nstep] += ((min_q - self.ucb_min[nstep]) * self.ucb_alpha)
        self.ucb_max[nstep] += ((max_q - self.ucb_max[nstep]) * self.ucb_alpha)
        self._diff[nstep] += ((diff - self._diff[nstep]) * self.ucb_alpha)

        # self.ucb_pref[nstep] += self.ucb_alpha * (-var_q - self.ucb_std[
            # nstep]) * (1 - self._ucb_prob(nstep))
        # for i in range(self._max_nstep):
            # if i != nstep:
                # self.ucb_pref[i] -= self.ucb_alpha * (-var_q - self.ucb_std[
                    # nstep]) * (-self._ucb_prob(i))
        
        

    def _ucb_nstep_for_indices(self, indices):
        if (np.any(self.ucb_second_nstep_count < 1) and
                self._nstep_distribution != "ucb_gradient_bandit"):

            start_range = self._ucb_total_steps % self._max_nstep + 1
            nsteps = np.tile(
                np.arange(start_range, self._max_nstep + 1),
                np.ceil(len(indices) / self._max_nstep))
            nsteps = nsteps[:len(indices)]
            for nstep in nsteps:
                self.ucb_nstep_count[nstep - 1] += 1
                self.ucb_second_nstep_count[nstep - 1] += 1
            self._ucb_total_steps += len(indices)
            return nsteps
        else:
            nsteps = []
            for _ in indices:
                if self._nstep_distribution != "ucb_gradient_bandit":
                    self.ucb = np.sqrt(2 * np.log(
                        self._ucb_total_steps) / self.ucb_nstep_count)

                if self._nstep_distribution == "ucb_classic":
                    ucb = self.ucb_mean + self.ucb
                elif self._nstep_distribution == "ucb_empirical":
                    ucb = self.ucb_mean + self.ucb_std
                elif self._nstep_distribution == "lcb_classic":
                    ucb = self.ucb_mean - self.ucb
                elif self._nstep_distribution == "lcb_empirical":
                    ucb = self.ucb_mean - self.ucb_std
                elif self._nstep_distribution == "lcb_min_clip":
                    ucb = self.ucb_min
                elif self._nstep_distribution == "ucb_max_clip":
                    ucb = self.ucb_max
                elif self._nstep_distribution == "ucb_gradient_bandit":
                    for i in range(self._max_nstep):
                        self.ucb_prob_cache[i] = self._ucb_prob(i)
                    ucb = self.ucb_prob_cache
                else:
                    raise ValueError("Unknown nstep_distribution: {}".format(
                        self._nstep_distribution))

                # nstep = np.argmax(ucb) + 1
                # take last occurrence of max
                nstep = len(ucb) - np.argmax(ucb[::-1])

                nsteps.append(nstep)
                self.ucb_nstep_count[nsteps[-1] - 1] += 1
                self.ucb_second_nstep_count[nsteps[-1] - 1] += 1
            self._ucb_total_steps += len(indices)
        if self._ucb_total_steps % 50000 == 0:
            self.ucb_second_nstep_count = np.zeros(self._max_nstep)
        return np.array(nsteps)

    def _sample_nstep_for_indices(self, indices):
        """For each index in `indices` sample a different n as its
        update_horizon. The n will be between 1 and self._max_nstep.
        """
        if (self._nstep_distribution is None
                or self._nstep_distribution == "fix"):
            return np.ones_like(indices) * self._update_horizon
        if self._nstep_distribution == "uniform":
            return np.random.randint(
                1, self._max_nstep + 1, size=len(indices))
        elif self._nstep_distribution == "beta":
            self.anneal_beta_distribution()
            pdf = np.random.beta(self._a, self._b, size=len(indices))
            return np.floor(pdf * self._max_nstep).astype(np.int32) + 1
        elif "ucb" in self._nstep_distribution:
            return self._ucb_nstep_for_indices(indices)
        elif "lcb" in self._nstep_distribution:
            return self._ucb_nstep_for_indices(indices)
        else:
            raise ValueError("Unknown nstep_distribution: {}".format(
                self._nstep_distribution))

    def load_dataset(self, dataset):
        n_transitions = dataset["observations"].shape[0]
        if n_transitions > self._replay_capacity:
            raise ValueError(
                "Replay buffer is smaller than the dataset to load!"
            )
        for i in range(n_transitions):
            obs = dataset['observations'][i]
            act = dataset['actions'][i]
            rew = dataset['rewards'][i]
            done = dataset['terminals'][i]
            self.add(obs, act, rew, done)

    @property
    def _episode_end_indices(self):
        logging.warning('The name `_episode_end_indices` will be deprecated '
                        'in a future version of Dopamine. Please use '
                        '`episode_end_indices` instead.')
        return self.episode_end_indices

    def _create_storage(self):
        """Creates the numpy arrays used to store transitions.
        """
        self._store = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            self._store[storage_element.name] = np.empty(
                array_shape, dtype=storage_element.type)

    def get_add_args_signature(self):
        """The signature of the add function.

        Note - Derived classes may return a different signature.

        Returns:
          list of ReplayElements defining the type of the argument signature needed
            by the add function.
        """
        return self.get_storage_signature()

    def get_storage_signature(self):
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          list of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement('observation', self._observation_shape,
                          self._observation_dtype),
            ReplayElement('action', self._action_shape, self._action_dtype),
            ReplayElement('reward', self._reward_shape, self._reward_dtype),
            ReplayElement('terminal', (), self._terminal_dtype)
        ]

        for extra_replay_element in self._extra_storage_types:
            storage_elements.append(extra_replay_element)
        return storage_elements

    def _add_zero_transition(self):
        """Adds a padding transition filled with zeros (Used in episode beginnings).
        """
        zero_transition = []
        for element_type in self.get_add_args_signature():
            zero_transition.append(
                np.zeros(element_type.shape, dtype=element_type.type))
        self.episode_end_indices.discard(self.cursor())  # If present
        self._add(*zero_transition)

    def add(self,
            observation,
            action,
            reward,
            terminal,
            *args,
            priority=None,
            episode_end=False):
        """Adds a transition to the replay memory.

        This function checks the types and handles the padding at the beginning of
        an episode. Then it calls the _add function.

        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.

        If the replay memory is at capacity the oldest transition will be discarded.

        Args:
          observation: np.array with shape observation_shape.
          action: int, the action in the transition.
          reward: float, the reward received in the transition.
          terminal: np.dtype, acts as a boolean indicating whether the transition
                    was terminal (1) or not (0).
          *args: extra contents with shapes and dtypes according to
            extra_storage_types.
          priority: float, unused in the circular replay buffer, but may be used
            in child classes like PrioritizedReplayBuffer.
          episode_end: bool, whether this experience is the last experience in
            the episode. This is useful for tasks that terminate due to time-out,
            but do not end on a terminal state. Overloading 'terminal' may not
            be sufficient in this case, since 'terminal' is passed to the agent
            for training. 'episode_end' allows the replay buffer to determine
            episode boundaries without passing that information to the agent.
        """
        if priority is not None:
            args = args + (priority,)

        # self._check_add_types(observation, action, reward, terminal, *args)
        if self._next_experience_is_episode_start:
            for _ in range(self._stack_size - 1):
                # Child classes can rely on the padding transitions being filled with
                # zeros. This is useful when there is a priority argument.
                self._add_zero_transition()
            self._next_experience_is_episode_start = False

        if episode_end or terminal:
            self.episode_end_indices.add(self.cursor())
            self._next_experience_is_episode_start = True
        else:
            self.episode_end_indices.discard(self.cursor())  # If present

        self._add(observation, action, reward, terminal, *args)

    def _add(self, *args):
        """Internal add method to add to the storage arrays.

        Args:
          *args: All the elements in a transition.
        """
        self._check_args_length(*args)
        transition = {e.name: args[idx]
                      for idx, e in enumerate(self.get_add_args_signature())}
        self._add_transition(transition)

    def _add_transition(self, transition):
        """Internal add method to add transition dictionary to storage arrays.

        Args:
          transition: The dictionary of names and values of the transition
                      to add to the storage.
        """
        cursor = self.cursor()
        for arg_name in transition:
            self._store[arg_name][cursor] = transition[arg_name]

        self.add_count += 1
        self.invalid_range = invalid_range(
            self.cursor(), self._replay_capacity, self._stack_size,
            self._update_horizon)

    def _check_args_length(self, *args):
        """Check if args passed to the add method have the same length as storage.

        Args:
          *args: Args for elements used in storage.

        Raises:
          ValueError: If args have wrong length.
        """
        if len(args) != len(self.get_add_args_signature()):
            raise ValueError('Add expects {} elements, received {}'.format(
                len(self.get_add_args_signature()), len(args)))

    def _check_add_types(self, *args):
        """Checks if args passed to the add method match those of the storage.

        Args:
          *args: Args whose types need to be validated.

        Raises:
          ValueError: If args have wrong shape or dtype.
        """
        self._check_args_length(*args)
        for arg_element, store_element in zip(args, self.get_add_args_signature()):
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
                # TODO(b/80536437). This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            else:
                # Assume it is scalar.
                arg_shape = tuple()
            store_element_shape = tuple(store_element.shape)
            if arg_shape != store_element_shape:
                raise ValueError('arg has shape {}, expected {}'.format(
                    arg_shape, store_element_shape))

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self.add_count == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self.add_count >= self._replay_capacity

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return self.add_count % self._replay_capacity

    def get_range(self, array, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Fast slice read when there is no wraparound.
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = array[start_index:end_index, ...]
        # Slow list read.
        else:
            indices = [(start_index + i) % self._replay_capacity
                       for i in range(end_index - start_index)]
            return_array = array[indices, ...]
        return return_array

    def get_observation_stack(self, index):
        return self._get_element_stack(index, 'observation')

    def _get_element_stack(self, index, element_name):
        state = self.get_range(self._store[element_name],
                               index - self._stack_size + 1, index + 1)
        # The stacking axis is 0 but the agent expects as the last axis.
        return np.moveaxis(state, 0, -1)

    def get_terminal_stack(self, index):
        return self.get_range(self._store['terminal'], index - self._stack_size + 1,
                              index + 1)

    def is_valid_transition(self, index, update_horizon=None):
        """Checks if the index contains a valid transition.

        Checks for collisions with the end of episodes and the current position
        of the cursor.

        Args:
          index: int, the index to the state in the transition.

        Returns:
          Is the index valid: Boolean.

        """
        if update_horizon is None:
            update_horizon = self._update_horizon

        # Check the index is in the valid range
        if index < 0 or index >= self._replay_capacity:
            return False
        if not self.is_full():
            # The indices and next_indices must be smaller than the cursor.
            if index >= self.cursor() - update_horizon:
                return False
            # The first few indices contain the padding states of the first episode.
            if index < self._stack_size - 1:
                return False

        # Skip transitions that straddle the cursor.
        if index in set(self.invalid_range):
            return False

        # If there are terminal flags in any other frame other than the last one
        # the stack is not valid, so don't sample it.
        if self.get_terminal_stack(index)[:-1].any():
            return False

        # If the episode ends before the update horizon, without a terminal signal,
        # it is invalid.
        for i in modulo_range(index, update_horizon, self._replay_capacity):
            if i in self.episode_end_indices and not self._store['terminal'][i]:
                return False

        return True

    def _create_batch_arrays(self, batch_size):
        """Create a tuple of arrays with the type of get_transition_elements.

        When using the WrappedReplayBuffer with staging enabled it is important to
        create new arrays every sample because StaginArea keeps a pointer to the
        returned arrays.

        Args:
          batch_size: (int) number of transitions returned. If None the default
            batch_size will be used.

        Returns:
          Tuple of np.arrays with the shape and type of get_transition_elements.
        """
        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = []
        for element in transition_elements:
            batch_arrays.append(np.empty(element.shape, dtype=element.type))
        return tuple(batch_arrays)

    def ready_to_sample(self):
        min_id = self._stack_size - 1
        max_id = self.cursor() - self._update_horizon
        return max_id > min_id

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = self.cursor() - self._replay_capacity + self._stack_size - 1
            max_id = self.cursor() - self._update_horizon
        else:
            # add_count < self._replay_capacity
            min_id = self._stack_size - 1
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:
                raise RuntimeError('Cannot sample a batch with fewer than stack size '
                                   '({}) + update_horizon ({}) transitions.'.
                                   format(self._stack_size, self._update_horizon))

        indices = []
        nsteps = []
        attempt_count = 0
        nstep = None
        while (len(indices) < batch_size and
               attempt_count < self._max_sample_attempts):
            index = np.random.randint(min_id, max_id) % self._replay_capacity
            if nstep is None:
                nstep = self._sample_nstep_for_indices([index])[0]
            if self.is_valid_transition(index, nstep):
                indices.append(index)
                nsteps.append(nstep)
            else:
                attempt_count += 1
        if len(indices) != batch_size:
            raise RuntimeError(
                'Max sample attempts: Tried {} times but only sampled {}'
                ' valid indices. Batch size is {}'.
                format(self._max_sample_attempts, len(indices), batch_size))

        return indices, nsteps

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the replay buffer.
        """
        nsteps = None
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices, nsteps = self.sample_index_batch(batch_size)
        assert len(indices) == batch_size
        if nsteps is None or len(nsteps) != batch_size:
            nsteps = self._sample_nstep_for_indices(indices)

        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = self._create_batch_arrays(batch_size)
        for batch_element, state_index in enumerate(indices):
            update_horizon = nsteps[batch_element]
            cum_discount_vector = np.array(
                [math.pow(self._gamma, n) for n in range(update_horizon)],
                dtype=np.float32)
            trajectory_indices = [(state_index + j) % self._replay_capacity
                                  for j in range(update_horizon)]
            trajectory_terminals = self._store['terminal'][trajectory_indices]
            is_terminal_transition = trajectory_terminals.any()
            if not is_terminal_transition:
                trajectory_length = update_horizon
            else:
                # np.argmax of a bool array returns the index of the first True.
                trajectory_length = np.argmax(
                    trajectory_terminals.astype(bool), 0) + 1
            next_state_index = state_index + trajectory_length
            trajectory_rewards = self.get_range(
                self._store['reward'], state_index, next_state_index)
            if self._mode == 'last':
                # cum_discount where all discounts are 0 except the last one
                cum_discount_vector[:-1] = 0.0
            elif self._mode == 'first':
                # cum_discount where all discounts are 0 except the first one
                cum_discount_vector[1:] = 0.0
            elif self._mode == 'max':
                # all values except where index is max_idx  set to zero
                max_idx = np.argmax(trajectory_rewards)
                cum_discount_vector[:max_idx] = 0.0
                cum_discount_vector[max_idx + 1:] = 0.0
            elif self._mode == 'min':
                # all values except where index is min_idx  set to zero
                min_idx = np.argmin(trajectory_rewards)
                cum_discount_vector[:min_idx] = 0.0
                cum_discount_vector[min_idx + 1:] = 0.0
            elif self._mode == 'mean':
                cum_discount_vector = np.ones_like(cum_discount_vector) / len(
                    cum_discount_vector)
            elif self._mode == 'reverse':
                cum_discount_vector = cum_discount_vector[::-1]
            elif self._mode == '':
                pass
            else:
                raise ValueError('Unknown mode: {}'.format(self._mode))

            trajectory_discount_vector = cum_discount_vector[:trajectory_length]

            # Fill the contents of each array in the sampled batch.
            assert len(transition_elements) == len(batch_arrays)
            for element_array, element in zip(batch_arrays, transition_elements):
                if element.name == 'state':
                    element_array[batch_element] = self.get_observation_stack(state_index)
                elif element.name == 'reward':
                    # compute the discounted sum of rewards in the trajectory.
                    element_array[batch_element] = np.sum(
                        trajectory_discount_vector * trajectory_rewards, axis=0)
                elif element.name == 'next_state':
                    element_array[batch_element] = self.get_observation_stack(
                        (next_state_index) % self._replay_capacity)
                elif element.name in ('next_action', 'next_reward'):
                    element_array[batch_element] = (
                        self._store[element.name.lstrip('next_')][(next_state_index) %
                                                                  self._replay_capacity])
                elif element.name == 'terminal':
                    element_array[batch_element] = is_terminal_transition
                elif element.name == 'indices':
                    element_array[batch_element] = state_index
                elif element.name == 'nstep':
                    element_array[batch_element] = update_horizon
                elif element.name in self._store.keys():
                    element_array[batch_element] = (
                        self._store[element.name][state_index])
                # We assume the other elements are filled in by the subclass.

        self._total_steps += 1
        return batch_arrays

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement('state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
            ReplayElement('action', (batch_size,) + self._action_shape,
                          self._action_dtype),
            ReplayElement('reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement('next_state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
            ReplayElement('next_action', (batch_size,) + self._action_shape,
                          self._action_dtype),
            ReplayElement('next_reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement('terminal', (batch_size,), self._terminal_dtype),
            ReplayElement('indices', (batch_size,), np.int32),
            ReplayElement('nstep', (batch_size,), np.int32)
        ]
        for element in self._extra_storage_types:
            transition_elements.append(
                ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                              element.type))
        return transition_elements

    def _generate_filename(self, checkpoint_dir, name, suffix):
        return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))

    def _return_checkpointable_elements(self):
        """Return the dict of elements of the class for checkpointing.

        Returns:
          checkpointable_elements: dict containing all non private (starting with
          _) members + all the arrays inside self._store.
        """
        checkpointable_elements = {}
        for member_name, member in self.__dict__.items():
            if member_name == '_store':
                for array_name, array in self._store.items():
                    checkpointable_elements[STORE_FILENAME_PREFIX + array_name] = array
            elif not member_name.startswith('_'):
                checkpointable_elements[member_name] = member
        return checkpointable_elements

    def save(self, checkpoint_dir, iteration_number):
        """Save the OutOfGraphReplayBuffer attributes into a file.

        This method will save all the replay buffer's state in a single file.

        Args:
          checkpoint_dir: str, the directory where numpy checkpoint files should be
            saved.
          iteration_number: int, iteration_number to use as a suffix in naming
            numpy checkpoint files.
        """
        pass

    def load(self, checkpoint_dir, suffix):
        """Restores the object from bundle_dictionary and numpy checkpoints.

        Args:
          checkpoint_dir: str, the directory where to read the numpy checkpointed
            files from.
          suffix: str, the suffix to use in numpy checkpoint files.

        Raises:
          NotFoundError: If not all expected files are found in directory.
        """
        pass


class SumTree(object):
    """A sum tree data structure for storing replay priorities.

    A sum tree is a complete binary tree whose leaves contain values called
    priorities. Internal nodes maintain the sum of the priorities of all leaf
    nodes in their subtree.

    For capacity = 4, the tree may look like this:

                 +---+
                 |2.5|
                 +-+-+
                   |
           +-------+--------+
           |                |
         +-+-+            +-+-+
         |1.5|            |1.0|
         +-+-+            +-+-+
           |                |
      +----+----+      +----+----+
      |         |      |         |
    +-+-+     +-+-+  +-+-+     +-+-+
    |0.5|     |1.0|  |0.5|     |0.5|
    +---+     +---+  +---+     +---+

    This is stored in a list of numpy arrays:
    self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5] ]

    For conciseness, we allocate arrays as powers of two, and pad the excess
    elements with zero values.

    This is similar to the usual array-based representation of a complete binary
    tree, but is a little more user-friendly.
    """

    def __init__(self, capacity):
        """Creates the sum tree data structure for the given replay capacity.

        Args:
          capacity: int, the maximum number of elements that can be stored in this
            data structure.

        Raises:
          ValueError: If requested capacity is not positive.
        """
        assert isinstance(capacity, int)
        if capacity <= 0:
            raise ValueError('Sum tree capacity should be positive. Got: {}'.
                             format(capacity))

        self.nodes = []
        tree_depth = int(math.ceil(np.log2(capacity)))
        level_size = 1
        for _ in range(tree_depth + 1):
            nodes_at_this_depth = np.zeros(level_size)
            self.nodes.append(nodes_at_this_depth)

            level_size *= 2

        self.max_recorded_priority = 1.0

    def _total_priority(self):
        """Returns the sum of all priorities stored in this sum tree.

        Returns:
          float, sum of priorities stored in this sum tree.
        """
        return self.nodes[0][0]

    def sample(self, query_value=None):
        """Samples an element from the sum tree.

        Each element has probability p_i / sum_j p_j of being picked, where p_i is
        the (positive) value associated with node i (possibly unnormalized).

        Args:
          query_value: float in [0, 1], used as the random value to select a
          sample. If None, will select one randomly in [0, 1).

        Returns:
          int, a random element from the sum tree.

        Raises:
          Exception: If the sum tree is empty (i.e. its node values sum to 0), or if
            the supplied query_value is larger than the total sum.
        """
        if self._total_priority() == 0.0:
            raise Exception('Cannot sample from an empty sum tree.')

        if query_value and (query_value < 0. or query_value > 1.):
            raise ValueError('query_value must be in [0, 1].')

        # Sample a value in range [0, R), where R is the value stored at the root.
        query_value = random.random() if query_value is None else query_value
        query_value *= self._total_priority()

        # Now traverse the sum tree.
        node_index = 0
        for nodes_at_this_depth in self.nodes[1:]:
            # Compute children of previous depth's node.
            left_child = node_index * 2

            left_sum = nodes_at_this_depth[left_child]
            # Each subtree describes a range [0, a), where a is its value.
            if query_value < left_sum:  # Recurse into left subtree.
                node_index = left_child
            else:  # Recurse into right subtree.
                node_index = left_child + 1
                # Adjust query to be relative to right subtree.
                query_value -= left_sum

        return node_index

    def stratified_sample(self, batch_size):
        """Performs stratified sampling using the sum tree.

        Let R be the value at the root (total value of sum tree). This method will
        divide [0, R) into batch_size segments, pick a random number from each of
        those segments, and use that random number to sample from the sum_tree. This
        is as specified in Schaul et al. (2015).

        Args:
          batch_size: int, the number of strata to use.
        Returns:
          list of batch_size elements sampled from the sum tree.

        Raises:
          Exception: If the sum tree is empty (i.e. its node values sum to 0).
        """
        if self._total_priority() == 0.0:
            raise Exception('Cannot sample from an empty sum tree.')

        bounds = np.linspace(0., 1., batch_size + 1)
        assert len(bounds) == batch_size + 1
        segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
        query_values = [random.uniform(x[0], x[1]) for x in segments]
        return [self.sample(query_value=x) for x in query_values]

    def get(self, node_index):
        """Returns the value of the leaf node corresponding to the index.

        Args:
          node_index: The index of the leaf node.
        Returns:
          The value of the leaf node.
        """
        return self.nodes[-1][node_index]

    def set(self, node_index, value):
        """Sets the value of a leaf node and updates internal nodes accordingly.

        This operation takes O(log(capacity)).
        Args:
          node_index: int, the index of the leaf node to be updated.
          value: float, the value which we assign to the node. This value must be
            nonnegative. Setting value = 0 will cause the element to never be
            sampled.

        Raises:
          ValueError: If the given value is negative.
        """
        if value < 0.0:
            raise ValueError('Sum tree values should be nonnegative. Got {}'.
                             format(value))
        self.max_recorded_priority = max(value, self.max_recorded_priority)

        delta_value = value - self.nodes[-1][node_index]

        # Now traverse back the tree, adjusting all sums along the way.
        for nodes_at_this_depth in reversed(self.nodes):
            # Note: Adding a delta leads to some tolerable numerical inaccuracies.
            nodes_at_this_depth[node_index] += delta_value
            node_index //= 2

        assert node_index == 0, ('Sum tree traversal failed, final node index '
                                 'is not 0.')


class PrioritisedMultiStepReplayBuffer(MultiStepReplayBuffer):
    """An out-of-graph Replay Buffer for Prioritized Experience Replay.

    See circular_replay_buffer.py for details.
    """

    def __init__(self,
                 observation_shape,
                 stack_size,
                 replay_capacity,
                 batch_size,
                 update_horizon=1,
                 nstep_distribution='beta',  # 'uniform', 'beta' or 'fix'
                 mode=False,
                 gamma=0.99,
                 epsilon=1e-10,
                 alpha=0.3,
                 beta=0.5,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32):
        """Initializes OutOfGraphPrioritizedReplayBuffer.

        Args:
          observation_shape: tuple of ints.
          stack_size: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          batch_size: int.
          update_horizon: int, length of update ('n' in n-step update).
          gamma: float, the discount factor.
          max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
          extra_storage_types: list of ReplayElements defining the type of the extra
            contents that will be stored and returned by sample_transition_batch.
          observation_dtype: np.dtype, type of the observations. Defaults to
            np.uint8 for Atari 2600.
          terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
            Atari 2600.
          action_shape: tuple of ints, the shape for the action vector. Empty tuple
            means the action is a scalar.
          action_dtype: np.dtype, type of elements in the action.
          reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
            means the reward is a scalar.
          reward_dtype: np.dtype, type of elements in the reward.
        """
        super(PrioritisedMultiStepReplayBuffer, self).__init__(
            observation_shape=observation_shape,
            stack_size=stack_size,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            update_horizon=update_horizon,
            nstep_distribution=nstep_distribution,
            mode=mode,
            gamma=gamma,
            max_sample_attempts=max_sample_attempts,
            extra_storage_types=extra_storage_types,
            observation_dtype=observation_dtype,
            terminal_dtype=terminal_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=reward_shape,
            reward_dtype=reward_dtype)

        self.sum_tree = SumTree(replay_capacity)
        self.eps = epsilon
        self.alpha = alpha
        self.beta = beta

    def get_add_args_signature(self):
        """The signature of the add function.

        The signature is the same as the one for OutOfGraphReplayBuffer, with an
        added priority.

        Returns:
          list of ReplayElements defining the type of the argument signature needed
            by the add function.
        """
        parent_add_signature = super(PrioritisedMultiStepReplayBuffer,
                                     self).get_add_args_signature()
        add_signature = parent_add_signature + [
            ReplayElement('priority', (), np.float32)
        ]
        return add_signature

    def _add(self, *args):
        """Internal add method to add to the underlying memory arrays.

        The arguments need to match add_arg_signature.

        If priority is none, it is set to the maximum priority ever seen.

        Args:
          *args: All the elements in a transition.
        """
        self._check_args_length(*args)

        # Use Schaul et al.'s (2015) scheme of setting the priority of new elements
        # to the maximum priority so far.
        # Picks out 'priority' from arguments and adds it to the sum_tree.
        transition = {}
        for i, element in enumerate(self.get_add_args_signature()):
            if element.name == 'priority':
                priority = args[i]
            else:
                transition[element.name] = args[i]

        self.sum_tree.set(self.cursor(), priority)
        super(PrioritisedMultiStepReplayBuffer, self)._add_transition(
            transition)

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled as in Schaul et al. (2015).

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          Exception: If the batch was not constructed after maximum number of tries.
        """
        # Sample stratified indices. Some of them might be invalid.
        indices = self.sum_tree.stratified_sample(batch_size)
        allowed_attempts = self._max_sample_attempts
        nsteps = []
        nstep = None
        for i in range(len(indices)):
            if nstep is None:
                nstep = self._sample_nstep_for_indices([indices[i]])[0]
            if not self.is_valid_transition(indices[i], nstep):
                if allowed_attempts == 0:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.
                        format(self._max_sample_attempts, i, batch_size))
                index = indices[i]
                while not self.is_valid_transition(
                        index, nstep) and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    index = self.sum_tree.sample()
                    # nstep = self._sample_nstep_for_indices([index])[0]
                    allowed_attempts -= 1
                indices[i] = index
            nsteps.append(nstep)
        return indices, nsteps

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions with extra storage and the priorities.

        The extra storage are defined through the extra_storage_types constructor
        argument.

        When the transition is terminal next_state_batch has undefined contents.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().
        """
        transition = (super(PrioritisedMultiStepReplayBuffer, self).
                      sample_transition_batch(batch_size, indices))
        transition_elements = self.get_transition_elements(batch_size)
        transition_names = [e.name for e in transition_elements]
        probabilities_index = transition_names.index('sampling_probabilities')
        indices_index = transition_names.index('indices')
        indices = transition[indices_index]
        # The parent returned an empty array for the probabilities. Fill it with the
        # contents of the sum tree.
        priorities = self.get_priority(indices)
        probs = priorities / self.sum_tree._total_priority()  # TODO: not sure if this already done or not.
        weights = (self._replay_capacity * probs) ** (-self.beta)
        # weights = priorities ** (-self.beta)
        weights /= weights.max()
        transition[probabilities_index][:] = weights
        return transition

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al.

        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
          priorities: float, the corresponding priorities.
        """
        assert indices.dtype == np.int32, ('Indices must be integers, '
                                           'given: {}'.format(indices.dtype))
        # Convert JAX arrays to NumPy arrays first, since it is faster to iterate
        # over the entirety of a NumPy array than a JAX array.

        priorities = priorities.cpu().numpy()
        priorities = (priorities + self.eps) ** self.alpha
        priorities = np.asarray(priorities)
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(index, priority)

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices.

        For any memory location not yet used, the corresponding priority is 0.

        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).

        Returns:
          priorities: float, the corresponding priorities.
        """
        assert indices.shape, 'Indices must be an array.'
        assert indices.dtype == np.int32, ('Indices must be int32s, '
                                           'given: {}'.format(indices.dtype))
        batch_size = len(indices)
        priority_batch = np.empty((batch_size), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self.sum_tree.get(memory_index)
        return priority_batch

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        parent_transition_type = (
            super(PrioritisedMultiStepReplayBuffer,
                  self).get_transition_elements(batch_size))
        probablilities_type = [
            ReplayElement('sampling_probabilities', (batch_size,), np.float32)
        ]
        return parent_transition_type + probablilities_type

