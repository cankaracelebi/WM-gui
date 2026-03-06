import os
import math
import random
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

from .base import Environment


class Orb:
    __slots__ = ("x", "y", "vx", "vy", "radius", "color", "glow_color")

    def __init__(self, size: int):
        self.radius = random.randint(3, 7)
        self.x = random.uniform(self.radius, size - self.radius)
        self.y = random.uniform(self.radius, size - self.radius)
        speed = random.uniform(0.3, 0.8)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.color = random.choice([
            (255, 50, 150),   # magenta
            (255, 200, 50),   # gold
            (100, 255, 80),   # lime
            (255, 140, 40),   # orange
            (180, 100, 255),  # violet
            (50, 200, 255),   # sky blue
        ])
        self.glow_color = tuple(min(255, c + 60) for c in self.color)

    def update(self, size: int):
        self.x += self.vx
        self.y += self.vy
        if self.x - self.radius < 0 or self.x + self.radius > size:
            self.vx = -self.vx
            self.x = max(self.radius, min(size - self.radius, self.x))
        if self.y - self.radius < 0 or self.y + self.radius > size:
            self.vy = -self.vy
            self.y = max(self.radius, min(size - self.radius, self.y))

    def draw(self, surface: pygame.Surface):
        pos = (int(self.x), int(self.y))
        # Glow halo
        pygame.draw.circle(surface, (*self.glow_color, 40), pos, self.radius + 3)
        pygame.draw.circle(surface, (*self.glow_color, 80), pos, self.radius + 1)
        # Core
        pygame.draw.circle(surface, self.color, pos, self.radius)
        # Bright center
        pygame.draw.circle(surface, (255, 255, 255), pos, max(1, self.radius // 3))


class Particle:
    __slots__ = ("x", "y", "bright", "timer")

    def __init__(self, size: int):
        self.x = random.randint(4, size - 4)
        self.y = random.randint(4, size - 4)
        self.bright = True
        self.timer = 0

    def update(self):
        self.timer += 1
        if self.timer % 4 == 0:
            self.bright = not self.bright

    def draw(self, surface: pygame.Surface):
        color = (220, 230, 255) if self.bright else (120, 140, 180)
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), 1)
        if self.bright:
            pygame.draw.circle(surface, (255, 255, 255, 100), (int(self.x), int(self.y)), 2)


class CosmicDriftEnv(Environment):
    """
    Cosmic Drift: A colorful space environment with glowing orbs and a player ship.

    Actions: 0=noop, 1=up, 2=down, 3=left, 4=right
    """

    ACTIONS = ["noop", "up", "down", "left", "right"]

    def __init__(self, size: int = 64, num_orbs: int = 5, num_particles: int = 8):
        self.size = size
        self.num_orbs = num_orbs
        self.num_particles = num_particles

        if not pygame.get_init():
            pygame.init()

        self.surface = pygame.Surface((size, size), pygame.SRCALPHA)
        self._bg = self._make_background()

        self.ship_x = 0.0
        self.ship_y = 0.0
        self.ship_speed = 1.2
        self.ship_size = 4
        self.orbs: list[Orb] = []
        self.particles: list[Particle] = []
        self.step_count = 0

        self.reset()

    def _make_background(self) -> pygame.Surface:
        """Dark space with subtle radial gradient."""
        bg = pygame.Surface((self.size, self.size))
        cx, cy = self.size // 2, self.size // 2
        max_dist = math.sqrt(cx * cx + cy * cy)
        for y in range(self.size):
            for x in range(self.size):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                t = dist / max_dist
                r = int(8 + 12 * (1 - t))
                g = int(8 + 18 * (1 - t))
                b = int(20 + 30 * (1 - t))
                bg.set_at((x, y), (r, g, b))
        # Scatter some dim stars
        for _ in range(30):
            sx, sy = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            brightness = random.randint(40, 90)
            bg.set_at((sx, sy), (brightness, brightness, brightness + 20))
        return bg

    def reset(self) -> np.ndarray:
        self.ship_x = self.size / 2.0
        self.ship_y = self.size / 2.0
        self.orbs = [Orb(self.size) for _ in range(self.num_orbs)]
        self.particles = [Particle(self.size) for _ in range(self.num_particles)]
        self.step_count = 0
        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        reward = 0.0
        self.step_count += 1

        # Move ship
        dx, dy = 0.0, 0.0
        if action == 1:
            dy = -self.ship_speed
        elif action == 2:
            dy = self.ship_speed
        elif action == 3:
            dx = -self.ship_speed
        elif action == 4:
            dx = self.ship_speed

        self.ship_x = (self.ship_x + dx) % self.size
        self.ship_y = (self.ship_y + dy) % self.size

        # Update orbs
        for orb in self.orbs:
            orb.update(self.size)
            # Check collision with ship
            dist = math.sqrt((orb.x - self.ship_x) ** 2 + (orb.y - self.ship_y) ** 2)
            if dist < orb.radius + self.ship_size:
                reward = -1.0
                self.ship_x = self.size / 2.0
                self.ship_y = self.size / 2.0

        # Update and check particles
        for particle in self.particles:
            particle.update()
            dist = math.sqrt((particle.x - self.ship_x) ** 2 + (particle.y - self.ship_y) ** 2)
            if dist < self.ship_size + 2:
                reward += 1.0
                particle.x = random.randint(4, self.size - 4)
                particle.y = random.randint(4, self.size - 4)

        frame = self.render()
        info = {"step": self.step_count, "ship_x": self.ship_x, "ship_y": self.ship_y}
        return frame, reward, False, info

    def render(self) -> np.ndarray:
        # Background
        self.surface.blit(self._bg, (0, 0))

        # Particles (behind orbs)
        for p in self.particles:
            p.draw(self.surface)

        # Orbs with glow
        for orb in self.orbs:
            orb.draw(self.surface)

        # Ship: bright cyan diamond with glow
        sx, sy = int(self.ship_x), int(self.ship_y)
        s = self.ship_size

        # Ship glow
        glow_surf = pygame.Surface((s * 4, s * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (0, 255, 255, 30), (s * 2, s * 2), s * 2)
        pygame.draw.circle(glow_surf, (0, 255, 255, 60), (s * 2, s * 2), s + 2)
        self.surface.blit(glow_surf, (sx - s * 2, sy - s * 2))

        # Ship body (diamond)
        points = [(sx, sy - s), (sx + s, sy), (sx, sy + s), (sx - s, sy)]
        pygame.draw.polygon(self.surface, (0, 240, 255), points)
        pygame.draw.polygon(self.surface, (180, 255, 255), [(sx, sy - s + 1), (sx + s - 1, sy), (sx, sy + s - 1), (sx - s + 1, sy)])

        # Engine trail (small fading dots behind ship)
        trail_color = (0, 180, 220)
        for i in range(1, 4):
            alpha = max(0, 150 - i * 50)
            trail_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (*trail_color, alpha), (1, 1), 1)
            self.surface.blit(trail_surf, (sx - 1, sy + s + i * 2))

        # Convert to numpy (H, W, 3) uint8
        frame = pygame.surfarray.array3d(self.surface)
        frame = frame.transpose(1, 0, 2)  # pygame uses (W, H) layout
        return frame.copy()

    @property
    def action_space_size(self) -> int:
        return 5

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self.size, self.size, 3)

    @property
    def action_names(self) -> list[str]:
        return self.ACTIONS
