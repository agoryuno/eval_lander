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

from stable_baselines3.common.evaluation import evaluate_policy
class EvalLander(LunarLander):

    def __init__(self, init_vals: Union[int, Union[List, Tuple]],
                 stabilize_terrain: bool = False,
                 *args, **kwargs):
        self.stabilize_terrain = stabilize_terrain
        self.__init_vals = init_vals
        if isinstance(init_vals, int):
            self.__init_vals = [
                (np.random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                 np.random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM))
                for i in range(init_vals)]

        self._next_init = (i for i in self.__init_vals)
        self.episodes_length = len(self.__init_vals)
        super().__init__(*args, **kwargs)

        self.__heights = []
        if self.stabilize_terrain:
            CHUNKS = 11
            H = VIEWPORT_H / SCALE
            for i in range(self.episodes_length):
                self.__heights.append(self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,)))
            self._next_heights = (i for i in self.__heights)

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
        for i, height in enumerate(self._next_heights):
            return i, height

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

        if self.stabilize_terrain:
            i, height = self.next_heights()
            if height is None:
                print(i, self.__heights[i])
        else:
            height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))

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
