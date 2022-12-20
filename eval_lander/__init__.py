import math
from typing import Union, List, Tuple, Optional
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


class EvalLander(LunarLander):

    def __init__(self, init_vals: Union[int, Union[List, Tuple]],
                 init_heights: Union[bool, Union[List, Tuple]] = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                math.pi,
                5.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        self.__init_vals = [init_vals]
        if isinstance(init_vals, int):
            self.__init_vals = [
                (np.random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                 np.random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM))
                for i in range(init_vals)]

        self._next_init = (i for i in self.__init_vals)
        self.episodes_length = len(self.__init_vals)
        self.episodes_num = self.episodes_length - 1

        self.__heights = []
        if init_heights is not None:
            self.__heights = [init_heights]
            if isinstance(init_heights, bool):
                self.__heights = self._make_heights(len(self.__init_vals))
            self.stabilize_terrain = True
            assert len(self.__heights) == len(self.__init_vals)

        self._next_heights = (i for i in self.__heights)

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
        self._next_init = (i for i in self.__init_vals)
        if self.stabilize_terrain:
            self._next_heights = (i for i in self.__heights)

    def next_init(self):
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
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # terrain
        CHUNKS = 11

        height = self.next_heights()
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
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
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
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
            #(
                #self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                #self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            #),
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

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]
