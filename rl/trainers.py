from collections import deque, Counter

from rl.buffers import OffPolicyReplayBuffer
from rl.algorithms import NaiveDQN
from rl.callbacks import CallbacksList
from tqdm.auto import tqdm
import numpy as np


class _LogHelper:
    def __init__(self, reward_window=200, loss_window=50, episode_window=20, action_window=500):
        self._reward_window = deque(maxlen=reward_window)
        self._loss_window = deque(maxlen=loss_window)
        self._ep_reward_window = deque(maxlen=episode_window)
        self._ep_length_window = deque(maxlen=episode_window)
        self._action_window = deque(maxlen=action_window)
        self._ep_reward = 0.0
        self._ep_length = 0

    def on_step(self, reward, done, action):
        self._reward_window.append(reward)
        self._action_window.append(int(action))
        self._ep_reward += reward
        self._ep_length += 1
        if done:
            self._ep_reward_window.append(self._ep_reward)
            self._ep_length_window.append(self._ep_length)
            self._ep_reward = 0.0
            self._ep_length = 0

    @staticmethod
    def _stats(prefix, values):
        arr = np.asarray(values, dtype=np.float32)
        return {
            f'{prefix}/mean': float(np.mean(arr)),
            f'{prefix}/min': float(np.min(arr)),
            f'{prefix}/max': float(np.max(arr)),
            f'{prefix}/std': float(np.std(arr)),
            f'{prefix}/median': float(np.median(arr)),
        }

    def get_stats(self, loss, q_values, agent, buffer):
        self._loss_window.append(loss)
        stats = {}

        if self._reward_window:
            stats.update(self._stats('reward', self._reward_window))
        if self._loss_window:
            stats.update(self._stats('loss', self._loss_window))
        if q_values is not None and len(q_values) > 0:
            stats.update(self._stats('q_values', q_values))
        if self._ep_reward_window:
            stats.update(self._stats('episode_reward', self._ep_reward_window))
            stats.update(self._stats('episode_length', self._ep_length_window))
        if self._action_window:
            total = len(self._action_window)
            for action, count in Counter(self._action_window).items():
                stats[f'actions/action_{action}_freq'] = count / total

        stats['train/epsilon'] = agent.epsilon
        stats['train/buffer_size'] = len(buffer.buffer)

        return stats


class Trainer:
    def __init__(self, env, agent, n_steps=8, batch_size=32, buffer_max_len=50000, callbacks=None):
        self.env = env
        self.agent = agent
        self.n_steps = n_steps
        self.batch_size = batch_size
        # ИЗМЕНЕНО: пустой CallbacksList по умолчанию вместо None
        self.callbacks = CallbacksList(callbacks if callbacks is not None else [])

        if isinstance(agent, NaiveDQN):
            self.buffer = OffPolicyReplayBuffer(max_len=buffer_max_len)
        else:
            raise TypeError(f'Ты чо дебил какой ещё {agent.__class__.__name__}, мне NaiveDQN/DQN/DDQN нужен!')

    def train(self, total_steps=10000, reward_window=200, loss_window=50):
        log = _LogHelper(reward_window=reward_window, loss_window=loss_window)

        state, _ = self.env.reset()
        episode = 0
        episode_step = 0
        mean_reward = 0.0
        mean_loss = 0.0

        pbar = tqdm(range(total_steps), desc='Agent learning')
        self.callbacks.on_training_start(self.agent, self.env)
        self.callbacks.on_episode_start(episode, self.agent, self.env)

        for step in pbar:
            action = self.agent.predict(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.update(state, next_state, action, reward, done)
            log.on_step(reward, done, action)

            self.callbacks.on_step(
                state, action, reward, next_state, done, info,
                step, episode, episode_step, self.agent, self.env
            )

            state = next_state
            episode_step += 1

            if done:
                self.callbacks.on_episode_end(
                    episode, log._ep_reward_window[-1] if log._ep_reward_window else 0,
                    log._ep_length_window[-1] if log._ep_length_window else 0,
                    self.agent, self.env
                )
                state, _ = self.env.reset()
                episode += 1
                episode_step = 0
                self.callbacks.on_episode_start(episode, self.agent, self.env)

            if (step + 1) % self.n_steps == 0:
                samples = self.buffer.sample(self.batch_size)
                loss, q_values = self.agent.update(*samples)
                stats = log.get_stats(loss, q_values, self.agent, self.buffer)
                mean_reward = stats.get('reward/mean', mean_reward)
                mean_loss = stats.get('loss/mean', mean_loss)
                self.callbacks.on_update(stats, step, self.agent, self.env)

            pbar.set_description(
                f'Reward: {mean_reward:.3f}; Loss: {mean_loss:.5f}; Epsilon: {self.agent.epsilon:.5f}'
            )

        self.callbacks.on_training_end()

    def demo(self, env, total_steps=1000):
        state, _ = env.reset()

        for _ in tqdm(range(total_steps), desc='Agent demonstration'):
            action = self.agent.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                state, _ = env.reset()

        if env.render_mode == 'human':
            env.close()
