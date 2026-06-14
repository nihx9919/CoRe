#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
from logger import Logger
from replay_buffer import ReplayBuffer
from collections import deque
import utils
import hydra
from omegaconf import OmegaConf
from collections import defaultdict
import importlib
import sys
import wandb
FRM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        utils.set_seed_everywhere(cfg.seed)

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        # make env
        self.log_success = False
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
        else:
            self.env = utils.make_env(cfg)

        # create agent
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            cfg.device,
            store_image=False,
            image_size=None)

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print("Using: {}".format(self.device))

        self.step = 0

        self.save_gif_dir = os.path.join(self.work_dir, 'eval_gifs')
        if not os.path.exists(self.save_gif_dir):
            os.makedirs(self.save_gif_dir)
        OmegaConf.save(config=self.cfg, f=f'{self.work_dir}/cfg.yaml')

        # import FRM
        try:
            sys.path.append(f"{self.cfg.work_dir_main}")
            FRM_file_name = f"env_iter{self.cfg.iter}_response{self.cfg.response}"
            FRM = importlib.import_module(FRM_file_name)
            self.frm = getattr(FRM, "reward_function")
            
            print(f"Using FRM file: {FRM_file_name}")
            print(f"Tensorboard Directory: {os.getcwd()}/tb")
        except:
            raise ImportError("Import FRM Error!")

        self.save_eval_gif = False
        self.use_wandb = cfg.use_wandb
        # for create wandb
        if self.use_wandb:
            group_name = f"{cfg.env}_FRM"
            wandb_name = f"{group_name}_seed{cfg.seed}"
            self.my_wandb = wandb.init(project="CoRe-frm-init",
                                       group=group_name,
                                       name=wandb_name,
                                       mode="online",
                                       )
        OmegaConf.save(config=self.cfg, f=f'{self.work_dir}/cfg.yaml')

    def evaluate(self):
        average_true_episode_reward = 0
        success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            print("evaluating episode {}".format(episode))
            images = []
            obs = self.env.reset()
            if "metaworld" in self.cfg.env:
                obs = obs[0]

            done = False
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                try:
                    obs, reward, done, extra = self.env.step(action)
                except:
                    obs, reward, terminated, truncated, extra = self.env.step(action)
                    done = terminated or truncated

                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
                if self.save_eval_gif:
                    if "metaworld" in self.cfg.env:
                        rgb_image = self.env.render()
    
                        rgb_image = rgb_image[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            rgb_image = rgb_image[100:400, 100:400, :]
                    else:
                        rgb_image = self.env.render(mode='rgb_array')

                    if 'softgym' not in self.cfg.env:
                        images.append(rgb_image)
                
            if self.save_eval_gif:
                if 'softgym' in self.cfg.env:
                    images = self.env.video_frames

            if self.save_eval_gif:
                # save gif image
                if self.log_success:
                    save_gif_path = os.path.join(self.save_gif_dir, 'step{:07}_episode{:02}_success{}_{}.gif'.format(self.step, episode, episode_success, round(true_episode_reward, 2)))
                else:
                    save_gif_path = os.path.join(self.save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(self.step, episode, round(true_episode_reward, 2)))
                utils.save_numpy_as_gif(np.array(images), save_gif_path)

            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_true_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)

        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
            self.logger.log('eval/success_rate', success_rate, self.step)
            if self.use_wandb:
                self.my_wandb.log({f"{self.cfg.env}": success_rate}, step=self.step)
                
        self.logger.dump(self.step, ty='eval')

    def run(self):
        model_save_dir = os.path.join(self.work_dir, "models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        episode, episode_reward, done = 0, 0, True
        true_episode_reward = 0
        if self.log_success:
            episode_success = 0
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)

        start_time = time.time()

        # for initialize wandb when step=0
        if self.use_wandb:
            self.my_wandb.log({f"{self.cfg.env}": 0}, step=0)
        
        while self.step <= self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()

                    # frm reward component logging
                    frm_reward_dict = defaultdict(float)
                    for d in frm_reward_hat_dict:
                        for key, value in d.items():
                            frm_reward_dict[key] += value
                    frm_reward_dict = dict(frm_reward_dict)
                    for key, value in frm_reward_dict.items():
                        self.logger.log('train/frm_' + key, value, self.step)

                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/true_episode_reward', true_episode_reward, self.step) 
                    if self.log_success:
                        self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/episode', episode, self.step)

                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps+1), ty='train')

                # evaluate agent periodically
                if self.step >= self.cfg.num_seed_steps and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                # reset env
                obs = self.env.reset()
                if "metaworld" in self.cfg.env:
                    obs = obs[0]
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                frm_reward_hat_dict = []

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.step > self.cfg.num_seed_steps:  
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            try: # for handle stupid gym wrapper change 
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
                
            # self.env.render()
            # allow infinite bootstrap
            done = float(done)
            if 'softgym' not in self.cfg.env:
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            else:
                done_no_max = done
            # get frm reward
            if "softgym_PassWater" in self.cfg.env:
                reward_hat, reward_hat_info = self.frm(next_obs, action, 1.2)
            elif "softgym" in self.cfg.env:
                reward_hat, reward_hat_info = self.frm(next_obs, action, None)
            else:
                reward_hat, reward_hat_info = self.frm(next_obs, action, self.env.target_pos_frm)

            frm_reward_hat_dict.append(reward_hat_info)

            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            # adding data to the reward training data
            self.replay_buffer.add(obs, action, reward_hat, 0, 0, 
                                   next_obs, done, done_no_max)


            obs = next_obs
            episode_step += 1
            self.step += 1
            
            if self.step % self.cfg.save_model_step == 0:
                self.agent.save(model_save_dir, self.step)
            

@hydra.main(config_path='config/frm_iter.yaml', strict=True)
def main(cfg):
    
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
