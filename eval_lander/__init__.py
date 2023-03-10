import math
from typing import Union, List, Tuple, Optional, Iterable
import numpy as np

from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

from gym import spaces
from gym.envs.box2d import LunarLander
from gym.envs.box2d.lunar_lander import (
    ContactDetector,
    VIEWPORT_H,
    VIEWPORT_W,
    SCALE,
    LANDER_POLY,
    LEG_AWAY,
    LEG_DOWN,
    LEG_W,
    LEG_H,
    LEG_SPRING_TORQUE,
    INITIAL_RANDOM
)


W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CHUNKS = 11
CHUNK_X = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
HELIPAD_X1 = CHUNK_X[CHUNKS // 2 - 1]
HELIPAD_X2 = CHUNK_X[CHUNKS // 2 + 1]
HELIPAD_Y = H / 4

class FTLander(LunarLander):

    def __init__(self, x: float, y: float, margin: float = 5.,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__x = x
        self.__y = y
        self.__margin = margin



    def next_init(self):
        return (np.random.uniform(self.__x-self.__margin, self.__x+self.__margin),
                np.random.uniform(self.__y-self.__margin, self.__y+self.__margin),)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super(LunarLander, self).reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        # terrain

        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))

        self.helipad_x1 = HELIPAD_X1
        self.helipad_x2 = HELIPAD_X2
        self.helipad_y = HELIPAD_Y
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (CHUNK_X[i], smooth_y[i])
            p2 = (CHUNK_X[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H / SCALE
        self.lander: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)
        self.lander.ApplyForceToCenter(
            self.next_init(),
            True,
        )

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}


class EvalLander(LunarLander):

    def __init__(self, init_vals: Optional[Union[int, Iterable[Tuple[float, float]]]] = None,
                 init_heights: Union[bool, int, Iterable[np.ndarray]] = False,
                 *args, **kwargs):
        if init_vals is None:
            assert isinstance(init_heights, int) or not init_heights, \
                "init_heights must be an int or False if init_vals is None"

        super().__init__(*args, **kwargs)

        self.__initial_random = INITIAL_RANDOM

        self.__init_vals = init_vals
        self.stabilize_forces = False
        self.episodes_length = None

        if self.__init_vals is not None:
            self.stabilize_forces = True
            if isinstance(init_vals, int):
                self.__init_vals = [
                    (np.random.uniform(-self.__initial_random, self.__initial_random),
                     np.random.uniform(-self.__initial_random, self.__initial_random))
                    for i in range(init_vals)]

            self._next_init = (i for i in self.__init_vals)
            self.episodes_length = len(self.__init_vals)
            self.episodes_num = self.episodes_length - 1

        if not self.episodes_length:
            if isinstance(init_heights, int):
                self.episodes_length = init_heights
                self.episodes_num = self.episodes_length - 1
        self.__heights = []
        if init_heights:
            self.__heights = init_heights
            if isinstance(init_heights, bool):
                self.__heights = self._make_heights(self.episodes_length)
            self.stabilize_terrain = True

        self._next_heights = (i for i in self.__heights)

    def _make_force(self):
        return (self.np_random.uniform(-self.__initial_random, self.__initial_random),
                self.np_random.uniform(-self.__initial_random, self.__initial_random))

    def _make_height(self):
        CHUNKS = 11
        H = VIEWPORT_H / SCALE
        return self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))

    def _make_heights(self, length):
        return [self._make_height() for i in range(length)]

    @property
    def heights(self):
        return self.__heights

    def reinit(self):
        if self.stabilize_forces:
            self._next_init = (i for i in self.__init_vals)
        if self.stabilize_terrain:
            self._next_heights = (i for i in self.__heights)

    def next_init(self):
        if not self.stabilize_forces:
            return self._make_force()
        for i in self._next_init:
            return i

    def next_heights(self):
        if not self.stabilize_terrain:
            return self._make_height()
        for height in self._next_heights:
            return height

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None
              ):
        super(LunarLander, self).reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        # terrain

        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))

        self.helipad_x1 = HELIPAD_X1
        self.helipad_x2 = HELIPAD_X2
        self.helipad_y = HELIPAD_Y
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (CHUNK_X[i], smooth_y[i])
            p2 = (CHUNK_X[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H / SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter(
            self.next_init(),
            True,
        )

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}
