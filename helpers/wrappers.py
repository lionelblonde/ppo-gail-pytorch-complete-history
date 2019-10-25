from collections import deque

import numpy as np
import cv2

import gym
from gym import spaces


cv2.ocl.setUseOpenCL(False)  # solves weird OpenCL X Python bindings conflicts


class LazyFrames(object):

    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers. This object should only be converted to numpy array before being passed to the
        model. You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Wrappers.

class PixelObsEnv(gym.Wrapper):

    def __init__(self, env, height, width, grayscale):
        super(PixelObsEnv, self).__init__(env)
        self.height = height
        self.width = width
        self.grayscale = grayscale
        self.num_colors = 1 if self.grayscale else 3
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(self.height,
                                                       self.width,
                                                       self.num_colors),
                                                dtype=np.uint8)

    def reset(self):
        self.env.reset()

        pixel_ob = self.env.render(height=self.height, width=self.width, mode='rgb_array')
        if self.grayscale:
            pixel_ob = cv2.cvtColor(pixel_ob, cv2.COLOR_RGB2GRAY)
            pixel_ob = np.expand_dims(pixel_ob, -1)

        return pixel_ob

    def step(self, ac):
        _, rew, done, infos = self.env.step(ac)

        pixel_ob = self.env.render(height=self.height, width=self.width, mode='rgb_array')
        if self.grayscale:
            pixel_ob = cv2.cvtColor(pixel_ob, cv2.COLOR_RGB2GRAY)
            pixel_ob = np.expand_dims(pixel_ob, -1)

        return pixel_ob, rew, done, infos


class FrameStack(gym.Wrapper):

    def __init__(self, env, k):
        """Stack k last frames. Returns lazy array, which is much more memory efficient."""
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(shp[:-1] + (shp[-1] * k,)),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
