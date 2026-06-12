# RL Learning

Библиотека алгоритмов обучения с подкреплением на PyTorch.

## Реализовано

### Алгоритмы (`rl/algorithms.py`)
| Класс | Описание |
|-------|----------|
| `NaiveDQN` | DQN без целевой сети |
| `DQN` | DQN с жёстким обновлением целевой сети |
| `DDQN` | Double DQN с мягким обновлением (усреднение Полякова) |

### Среды (`rl/envs.py`)
- `GridWorldEnv` — агент на сетке NxN, дойти до цели в правом нижнем углу, опциональный враг со случайным движением
- `SpaceAsteroidsEnv` — корабль уклоняется от астероидов, летящих справа налево; скорость и плотность астероидов растут со временем

### Инфраструктура
- `OffPolicyReplayBuffer` — буфер воспроизведения опыта (`rl/buffers.py`)
- `Trainer` — цикл обучения с логированием (`rl/trainers.py`)
- Система колбэков: `BaseCallback`, `CallbacksList`, `WandBCallback` (`rl/callbacks.py`)

## Использование

```python
from rl.algorithms import DDQN
from rl.envs import SpaceAsteroidsEnv
from rl.trainers import Trainer

env = SpaceAsteroidsEnv()
agent = DDQN(env, lr=5e-4, gamma=0.99, epsilon_decay_rate=0.99999)
trainer = Trainer(env, agent, batch_size=256, buffer_max_len=100_000)
trainer.train(total_steps=500_000)
```

Сохранение и загрузка:
```python
agent.save("agent.pt")
agent = DDQN(env, lr=5e-4, load_path="agent.pt")  # веса загружены, гиперпараметры переопределены
```

## Установка

```bash
pip install git+https://github.com/Bobby-Shmurda31-ML/rl.git
```

Или локально:
```bash
git clone https://github.com/Bobby-Shmurda31-ML/rl.git
cd rl
pip install -e .
```

## Планы
- [x] NaiveDQN / DQN / DDQN
- [ ] A2C
- [ ] PPO
- [ ] DDPG / TD3 / SAC
- [ ] DreamerV3 / TD-MPC2
