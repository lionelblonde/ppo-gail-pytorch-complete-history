# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MuJoCo environments.

MUJOCO_ROBOTS = [
    'InvertedPendulum',
    'InvertedDoublePendulum',
    'Reacher',
    'Hopper',
    'HalfCheetah',
    'Walker2d',
    'Ant',
    'Humanoid',
]

MUJOCO_ENVS = ["{}-v2".format(name) for name in MUJOCO_ROBOTS]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Atari games.

ATARI_GAMES = list(map(lambda name: ''.join([g.capitalize() for g in name.split('_')]), [
    'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout',
    'carnival', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon',
]))

ATARI_ENVS = ["{}NoFrameskip-v4".format(name) for name in ATARI_GAMES]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PyColab environments.

PYCOLAB_ENVS = [
    'BoxWorld',
    'CliffWalk',
]

PYCOLAB_ENVS = ["{}-v0".format(name) for name in PYCOLAB_ENVS]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Aggregate the environments

BENCHMARKS = {
    'mujoco': MUJOCO_ENVS,
    'atari': ATARI_ENVS,
    'pycolab': PYCOLAB_ENVS,
}
