import random
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, size=5, max_steps=50, render_mode=None, with_enemy=True):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.with_enemy = with_enemy

        self.action_space = spaces.Discrete(4)

        if self.with_enemy:
            self.observation_space = spaces.MultiDiscrete(
                [size, size, size, size], dtype=np.int32
            )
            self.enemy_pos = np.array([size // 2, size // 2], dtype=np.int32)
        else:
            self.observation_space = spaces.MultiDiscrete(
                [size, size], dtype=np.int32
            )
            self.enemy_pos = None

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([size - 1, size - 1], dtype=np.int32)
        self.current_step = 0

        self.cell_size = 100
        self.window_size = self.size * self.cell_size
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        if self.with_enemy:
            self.enemy_pos = np.array(
                [self.size // 2, self.size // 2], dtype=np.int32
            )
        self.current_step = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def _get_obs(self):
        if self.with_enemy:
            return np.concatenate([self.agent_pos, self.enemy_pos]).ravel()
        return self.agent_pos.copy().ravel()

    def step(self, action):
        if action == 0:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        hit_enemy = False
        if self.with_enemy:
            enemy_action = random.randint(0, 3)
            if enemy_action == 0:
                self.enemy_pos[1] = max(0, self.enemy_pos[1] - 1)
            elif enemy_action == 1:
                self.enemy_pos[1] = min(self.size - 1, self.enemy_pos[1] + 1)
            elif enemy_action == 2:
                self.enemy_pos[0] = max(0, self.enemy_pos[0] - 1)
            elif enemy_action == 3:
                self.enemy_pos[0] = min(self.size - 1, self.enemy_pos[0] + 1)
            hit_enemy = np.array_equal(self.agent_pos, self.enemy_pos)

        self.current_step += 1
        reached_goal = np.array_equal(self.agent_pos, self.goal_pos)

        terminated = bool(hit_enemy or reached_goal)
        truncated = bool(self.current_step >= self.max_steps)

        reward = -0.01

        if reached_goal:
            reward += 10.0
        elif hit_enemy:
            reward -= 10.0
        else:
            reward -= 1.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, x * self.cell_size),
                (self.window_size, x * self.cell_size),
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size),
            )

        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                self.goal_pos * self.cell_size,
                (self.cell_size, self.cell_size),
            ),
        )

        if self.with_enemy:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (self.enemy_pos * self.cell_size + self.cell_size // 2),
                self.cell_size // 3,
            )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_pos * self.cell_size + self.cell_size // 2),
            self.cell_size // 3,
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class SpaceAsteroidsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, width=800, height=600, max_steps=1000, max_asteroids_in_obs=10, render_mode=None):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.max_asteroids_in_obs = max_asteroids_in_obs

        self.action_space = spaces.Discrete(5)
        # ИЗМЕНЕНО: +1 для нормализованного шага
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5 + max_asteroids_in_obs * 4,), dtype=np.float32
        )

        self.max_agent_speed = 10.0
        self.max_asteroid_speed = 15.0

        self.thrust = 0.5
        self.friction = 0.99
        self.agent_radius = 15
        self.asteroid_radius = 20

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = np.array([100.0, self.height / 2.0], dtype=np.float32)
        self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.asteroids = []
        self.spawn_cooldown = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def _get_obs(self):
        norm_agent_x = (self.agent_pos[0] / self.width) * 2.0 - 1.0
        norm_agent_y = (self.agent_pos[1] / self.height) * 2.0 - 1.0
        norm_agent_vx = np.clip(self.agent_vel[0] / self.max_agent_speed, -1.0, 1.0)
        norm_agent_vy = np.clip(self.agent_vel[1] / self.max_agent_speed, -1.0, 1.0)

        # ИЗМЕНЕНО: добавлен нормализованный шаг
        norm_step = self.current_step / self.max_steps
        obs = [norm_agent_x, norm_agent_y, norm_agent_vx, norm_agent_vy, norm_step]

        sorted_asteroids = sorted(
            self.asteroids,
            key=lambda ast: np.linalg.norm(ast["pos"] - self.agent_pos)
        )

        for i in range(self.max_asteroids_in_obs):
            if i < len(sorted_asteroids):
                ast = sorted_asteroids[i]
                dx = ast["pos"][0] - self.agent_pos[0]
                dy = ast["pos"][1] - self.agent_pos[1]

                norm_dx = np.clip(dx / self.width, -1.0, 1.0)
                norm_dy = np.clip(dy / self.height, -1.0, 1.0)
                norm_vx = np.clip(ast["vel"][0] / self.max_asteroid_speed, -1.0, 1.0)
                norm_vy = np.clip(ast["vel"][1] / self.max_asteroid_speed, -1.0, 1.0)

                obs.extend([norm_dx, norm_dy, norm_vx, norm_vy])
            else:
                obs.extend([1.0, 1.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        if action == 1:
            self.agent_vel[1] -= self.thrust
        elif action == 2:
            self.agent_vel[1] += self.thrust
        elif action == 3:
            self.agent_vel[0] -= self.thrust
        elif action == 4:
            self.agent_vel[0] += self.thrust

        self.agent_vel *= self.friction
        self.agent_vel = np.clip(self.agent_vel, -self.max_agent_speed, self.max_agent_speed)
        self.agent_pos += self.agent_vel

        base_speed = 3.0 + (self.current_step * 0.005)
        speed_variance = base_speed * (0.2 + (self.current_step * 0.001))
        spawn_rate = max(5, 20 - self.current_step // 50)

        self.spawn_cooldown -= 1
        if self.spawn_cooldown <= 0:
            self.spawn_cooldown = spawn_rate
            vel_x = -random.uniform(
                base_speed - speed_variance, base_speed + speed_variance
            )
            vel_y = random.uniform(-1.0, 1.0)
            self.asteroids.append(
                {
                    "pos": np.array([self.width, random.uniform(0, self.height)], dtype=np.float32),
                    "vel": np.array([vel_x, vel_y], dtype=np.float32),
                }
            )

        for ast in self.asteroids:
            ast["pos"] += ast["vel"]

        self.asteroids = [
            ast for ast in self.asteroids
            if ast["pos"][0] > -self.asteroid_radius
               and -self.asteroid_radius < ast["pos"][1] < self.height + self.asteroid_radius
        ]

        hit_asteroid = False
        proximity_penalty = 0.0
        min_collision_dist = self.agent_radius + self.asteroid_radius

        for ast in self.asteroids:
            dist = np.linalg.norm(self.agent_pos - ast["pos"])

            if dist < min_collision_dist:
                hit_asteroid = True

            dist_edge = max(0.0, dist - min_collision_dist)
            if dist_edge < 150.0:
                proximity_penalty += 2.0 * np.exp(-0.04 * dist_edge)

        proximity_penalty = min(proximity_penalty, 5.0)

        hit_wall = (
                self.agent_pos[0] < self.agent_radius
                or self.agent_pos[0] > self.width - self.agent_radius
                or self.agent_pos[1] < self.agent_radius
                or self.agent_pos[1] > self.height - self.agent_radius
        )

        terminated = bool(hit_asteroid or hit_wall)
        truncated = bool(self.current_step >= self.max_steps)

        # ИЗМЕНЕНО: добавлена растущая награда за выживание
        reward = 0.1 + self.current_step / self.max_steps * 3.0 - proximity_penalty
        if terminated:
            reward = -100.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Space Asteroids")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((10, 10, 20))

        pygame.draw.rect(canvas, (100, 0, 0), canvas.get_rect(), 5)

        for ast in self.asteroids:
            pygame.draw.circle(canvas, (120, 120, 120), ast["pos"].astype(int), self.asteroid_radius)

        pygame.draw.circle(canvas, (0, 150, 255), self.agent_pos.astype(int), self.agent_radius)
        if np.linalg.norm(self.agent_vel) > 0.1:
            dir_vec = self.agent_vel / np.linalg.norm(self.agent_vel)
            nose_pos = self.agent_pos + dir_vec * self.agent_radius
            pygame.draw.line(canvas, (255, 255, 255), self.agent_pos.astype(int), nose_pos.astype(int), 3)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()