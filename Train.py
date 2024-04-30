import gym
import numpy as np
from RandomAgent import TimeLimitWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import os
import retro

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

def make_env(env_id, rank, seed=0):
    def _init():
        env = retro.make(game=env_id)
        env = TimeLimitWrapper(env, max_steps=2000)
        env = MaxAndSkipEnv(env, 4)
        env.seed(seed + rank)    #different exploration trajectory
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "MegaMan2-Nes"
    num_cpu = 4  # Number of processes to use
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]),"tmp/TestMonitor")

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003)

    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=5000000, callback=callback, tb_log_name="PPO-00003")
    model.save(env_id)
    print("------------- Done Learning -------------")
    env = retro.make(game=env_id)
    env = TimeLimitWrapper(env)
    
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()