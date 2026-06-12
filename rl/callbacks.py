from abc import ABC
import wandb

class BaseCallback(ABC):
    def on_training_start(self, agent, env, **kwargs):
        pass

    def on_training_end(self):
        pass

    def on_episode_start(self, episode, agent, env, **kwargs):
        pass

    def on_step(self, state, action, reward, next_state, done, info, step, episode,
                episode_step, agent, env, **kwargs):
        pass

    def on_episode_end(self, episode, total_reward, episode_length, agent, env, **kwargs):
        pass

    def on_update(self, stats: dict, step: int, agent, env, **kwargs):
        pass

    def on_rollout_end(self, buffer, agent):
        pass


class CallbacksList:
    def __init__(self, callbacks: list):
        self.callbacks = callbacks

    def __getattr__(self, name):
        def proxy(*args, **kwargs):
            for cb in self.callbacks:
                getattr(cb, name)(*args, **kwargs)
        return proxy


class WandBCallback(BaseCallback):
    def __init__(self, agent, project):
        self.run = wandb.init(
            project=project,
            config=agent.config
        )

    def on_update(self, stats, step, agent, env, **kwargs):
        self.run.log(stats, step=step)