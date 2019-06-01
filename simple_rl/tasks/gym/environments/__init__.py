from simple_rl.tasks.gym.environments.ant_maze_env import AntMazeEnv
from simple_rl.tasks.gym.environments.point_maze_env import PointMazeEnv

from gym.envs.registration import register

register(
    id='AntMaze-1-v0',
    entry_point='simple_rl.tasks.gym.environments.ant_maze_env:AntMazeEnv',
    kwargs={'maze_id': 'Maze', 'goal_pos': [16.0, 0.0]},
)

register(
    id='AntMaze-2-v0',
    entry_point='simple_rl.tasks.gym.environments.ant_maze_env:AntMazeEnv',
    kwargs={'maze_id': 'Maze', 'goal_pos': [16.0, 16.0]},
)

register(
    id='AntMaze-3-v0',
    entry_point='simple_rl.tasks.gym.environments.ant_maze_env:AntMazeEnv',
    kwargs={'maze_id': 'Maze', 'goal_pos': [0.0, 16.0]},
)


register(
    id='PointMaze-1-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Maze', 'goal_pos': [16.0, 0.0]},
)
register(
    id='PointMaze-2-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Maze', 'goal_pos': [16.0, 16.0]},
)
register(
    id='PointMaze-3-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Maze', 'goal_pos': [0.0, 16.0]},
)

register(
    id='PointMaze-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Maze', 'goal_pos': [0.0, 16.0]},
)


register(
    id='PointPush-1-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Push', 'goal_pos': [-8.0, 8.0]},
)
register(
    id='PointPush-2-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Push', 'goal_pos': [8.0, 8.0]},
)
register(
    id='PointPush-3-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Push', 'goal_pos': [0.0, 19.0]},
)

register(
    id='PointFall-1-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Fall', 'goal_pos': [8.0, 8.0]},
)
register(
    id='PointFall-2-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Fall', 'goal_pos': [8.0, 27.0]},
)
register(
    id='PointFall-3-v0',
    entry_point='simple_rl.tasks.gym.environments.point_maze_env:PointMazeEnv',
    kwargs={'maze_id': 'Fall', 'goal_pos': [0.0, 27.0]},
)
