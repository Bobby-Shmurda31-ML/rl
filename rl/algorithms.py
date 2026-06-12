import random
import torch.nn as nn
import numpy as np
import torch
import copy


class NN(nn.Module):
    def __init__(self, input_shape, net_arch):
        super(NN, self).__init__()

        net_arch = [input_shape[0]] + net_arch.copy()

        self.layers = nn.Sequential()
        for i in range(len(net_arch) - 1):
            self.layers.extend([
                nn.Linear(net_arch[i], net_arch[i + 1]),
                (nn.ReLU() if i != len(net_arch) - 2 else nn.Identity())
            ])

    def forward(self, x):
        return self.layers(x)


class NaiveDQN:
    def __init__(self, env, lr=4e-5, gamma=0.95, start_epsilon=1, epsilon_decay_rate=0.9999,
                 min_epsilon=0.05, max_grad_norm=1, net=None, device=None, load_path=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.max_grad_norm = max_grad_norm

        self.config = {
            'lr': lr,
            'gamma': gamma,
            'start_epsilon': start_epsilon,
            'min_epsilon': min_epsilon,
            'epsilon_decay_rate': epsilon_decay_rate,
            'max_grad_norm': max_grad_norm
        }

        if isinstance(net, list) or net is None:
            self.net_arch = ([32, 32] if net is None else net) + [env.action_space.n]
            self.model = NN(env.observation_space._shape, self.net_arch)
        else:
            self.net_arch = None
            self.model = net

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()

        if load_path is not None:
            self._load(load_path)

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def _load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])

    def get_next_q_values(self, next_states):
        next_q_values = self.model(next_states)
        next_actions = torch.argmax(next_q_values, dim=-1).unsqueeze(-1)
        next_q_values = next_q_values.gather(index=next_actions, dim=-1)
        return next_q_values

    def update(self, states, next_states, actions, rewards, dones):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states)

        with torch.no_grad():
            next_q_values = self.get_next_q_values(next_states)
            targets = rewards + self.gamma * (1 - dones) * next_q_values

        q_values_taken = q_values.gather(index=actions, dim=-1)
        loss = self.loss_fn(q_values_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        return loss.item(), q_values_taken.detach().cpu().numpy().flatten()

    def predict(self, state, deterministic=False):
        if self.epsilon > random.uniform(0, 1) and not deterministic:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            q_values = q_values.cpu().numpy()
            action = np.argmax(q_values, axis=-1)[0]

        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        return action


class _DQNWithTargetNN(NaiveDQN):
    def __init__(self, *args, **kwargs):
        load_path = kwargs.pop('load_path', None)
        super().__init__(*args, **kwargs)

        if self.net_arch is not None:
            self.target_model = NN(self.env.observation_space._shape, self.net_arch)
        else:
            self.target_model = copy.deepcopy(self.model)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)

        if load_path is not None:
            self._load(load_path)

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def _load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(data['model'])
        self.target_model.load_state_dict(data['target_model'])
        self.optimizer.load_state_dict(data['optimizer'])

    def _target_update(self):
        raise NotImplementedError('Этот класс должен быть дочерним классом DQN/DDQN.')

    def get_next_q_values(self, next_states):
        raise NotImplementedError('Этот класс должен быть дочерним классом DQN/DDQN.')

    def update(self, *args, **kwargs):
        loss, q_vals = super().update(*args, **kwargs)
        self._target_update()
        return loss, q_vals


class DQN(_DQNWithTargetNN):
    def __init__(self, *args, target_update_interval=100, **kwargs):
        super().__init__(*args, **kwargs)

        self.config['target_update_interval'] = target_update_interval
        self.target_update_interval = target_update_interval
        self.step = 0

    def get_next_q_values(self, next_states):
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_actions = torch.argmax(next_q_values, dim=-1).unsqueeze(-1)
        next_q_values = next_q_values.gather(index=next_actions, dim=-1)
        return next_q_values

    def _target_update(self):
        self.step += 1
        if self.step % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())


class DDQN(_DQNWithTargetNN):
    def __init__(self, *args, tau=0.005, **kwargs):
        super().__init__(*args, **kwargs)

        self.config['tau'] = tau
        self.tau = tau

    def get_next_q_values(self, next_states):
        with torch.no_grad():
            main_next_q_values = self.model(next_states)
            target_next_q_values = self.target_model(next_states)
        main_next_actions = torch.argmax(main_next_q_values, dim=-1).unsqueeze(-1)
        next_q_values = target_next_q_values.gather(index=main_next_actions, dim=-1)
        return next_q_values

    def _target_update(self):
        for target_param, main_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + main_param * self.tau)