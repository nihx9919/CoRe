#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
from logger import Logger
from replay_buffer import ReplayBuffer
from RRM.reward_model import RewardModel
from collections import deque
import utils
import hydra
import cv2
from omegaconf import OmegaConf
import wandb
import sys
from FRM.reward_code import FRMCode
from collections import defaultdict
from FRM.misc import file_to_string
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        utils.set_seed_everywhere(cfg.seed)

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name
            )

        # make env
        self.log_success = False
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
        else:
            raise ValueError(f"not support the env {self.env}")

        # create agent
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # set image size
        image_height = image_width = cfg.image_size 
        self.resize_factor = 1
        if "Rope" in cfg.env:
            image_height = image_width = 240
            self.resize_factor = 3
        elif "Water" in cfg.env:
            image_height = image_width = 360
            self.resize_factor = 2
        if "Cloth" in cfg.env:
            image_height = image_width = 360
        
        # set to 0 to make sure that rrm is 0 at the begining of the train
        self.reward_alpha = 0.0
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity), # we cannot afford to store too many images in the replay buffer.
            cfg.device,
            image_size=image_height, 
            reward_alpha=self.reward_alpha)

        # instantiating the reward model
        self.RRM = RewardModel(
            ### original PEBBLE parameters
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            capacity=cfg.max_feedback,
            
            ### vlm parameters
            env_name=cfg.env,
            log_dir=self.work_dir,
            cached_label_path=cfg.cached_label_path,

            ### image-based reward model parameters
            image_height=image_height,
            image_width=image_width,
            resize_factor=self.resize_factor,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,
        )
        
        if cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(cfg.reward_model_load_dir))
            self.RRM.load(cfg.reward_model_load_dir, 1000000)
        if cfg.agent_model_load_dir != "None":
            print("loading agent model at {}".format(cfg.agent_model_load_dir))
            self.agent.load(cfg.agent_model_load_dir, 1000000)

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print(f"Using: {self.device}")

        self.image_height = image_height
        self.image_width = image_width

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        self.save_eval_video = cfg.save_eval_video
        self.use_wandb = cfg.use_wandb

        if self.save_eval_video:
            self.save_eval_video_dir = os.path.join(self.work_dir, 'eval_videos')
            if not os.path.exists(self.save_eval_video_dir):
                os.makedirs(self.save_eval_video_dir)

        if self.use_wandb:
            group_name = f"{cfg.env}"
            wandb_name = f"{group_name}_seed{cfg.seed}"
            self.my_wandb = wandb.init(project="CoRe",
                                       group=group_name,
                                       name=wandb_name,
                                       mode="online",
                                       )
            
        # save config and prompt information
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        os.system("cp {}/RRM/prompt.py {}/".format(current_file_path, self.work_dir))
        OmegaConf.save(config=self.cfg, f=f'{self.work_dir}/cfg.yaml')

        # generate formal reward code
        sys.path.append(f"{self.work_dir}")
        sys.path.append(f"{PROJECT_DIR}/envs/env_info")
        sys.path.append(f"{PROJECT_DIR}/envs/frm_code")
        self.FRM_gen = FRMCode(cfg, PROJECT_DIR, self.work_dir, self.RRM, llm_model='gpt-4.1-mini', sample_num=4, once_sample_num=4)
        
        self.use_FRM_online = cfg.use_FRM_online
        # initialize the first FRM 
        if self.use_FRM_online:
            self.FRM = self.FRM_gen.reward_gen()
            if self.FRM is None:
                raise ValueError("FRM initialization fail!")
        else:
            # load FRM reward from local file
            FRM_reward_file = f"{self.cfg.env.replace('-', '_').lower()}_frm"
            self.FRM = self.FRM_gen.dynamic_import_frm(FRM_reward_file)
            self.FRM_gen.best_reward_response = file_to_string(f"{PROJECT_DIR}/envs/frm_code/{self.cfg.env.replace('-', '_').lower()}_frm.py")
        
        # empty preference dataset in RAM after reward model stop update
        self.empty_pre_dataset = True


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
                if "Cloth" in self.cfg.env:
                    true_episode_reward = reward
                else:
                    true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

                if self.save_eval_video:
                    if "metaworld" in self.cfg.env:
                        rgb_image = self.env.render()

                        rgb_image = rgb_image[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            rgb_image = rgb_image[100:400, 100:400, :]
                    else:
                        rgb_image = self.env.render(mode='rgb_array')

                    if 'softgym' not in self.cfg.env:
                        images.append(rgb_image)

            if self.save_eval_video:
                if 'softgym' in self.cfg.env:
                    images = self.env.video_frames
            
            if self.save_eval_video:
            # save video
                if self.log_success:
                    save_video_path = os.path.join(self.save_eval_video_dir, 'step{:07}_episode{:02}_success{}_{}.mp4'.format(self.step, episode, episode_success, round(true_episode_reward, 2)))
                else:
                    save_video_path = os.path.join(self.save_eval_video_dir, 'step{:07}_episode{:02}_{}.mp4'.format(self.step, episode, round(true_episode_reward, 2)))
                utils.save_list_as_mp4(images, save_video_path)

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
    
    def learn_reward(self, first_flag=0):
        # get feedbacks
        query_label_timer = time.time()
        labeled_queries = self.RRM.uniform_sampling()
        self.total_feedback += self.RRM.mb_size
        self.labeled_feedback += labeled_queries
        print(f'Query label time: {time.time() - query_label_timer}')
        self.logger.log('train_time/query_label', time.time() - query_label_timer, self.step)

        total_acc = 0
        reward_learning_timer = time.time()
        if self.labeled_feedback > 0:
            # update reward
            for _ in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    raise NotImplementedError
                else:
                    self.RRM.train()
                    train_acc = self.RRM.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break
        self.logger.log('train_time/reward_learning', time.time() - reward_learning_timer, self.step)
        print(f"Reward function is updated! ACC: {total_acc:.3f} Queried/Total: {self.labeled_feedback}/{self.total_feedback}")
        print(f'Reward learning time: {time.time() - reward_learning_timer}')
        return total_acc, self.RRM.vlm_label_acc

    def relabel_with_reward_model(self):
        self.RRM.eval()
        self.replay_buffer.relabel_with_predictor(self.RRM, self.step)
        self.RRM.train()

    def run(self):
        model_save_dir = os.path.join(self.work_dir, "models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        episode, episode_reward, done = 0, 0, True
        episode_end_reward = 0
        episode_reward_frm, episode_reward_rrm = 0, 0
        true_episode_reward = 0
        if self.log_success:
            episode_success = 0

        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)

        start_time = time.time()
        # for initialize wandb when step=0
        if self.use_wandb:
            self.my_wandb.log({f"{self.cfg.env}": 0}, step=0)

        interact_count = 0
        reward_learning_acc = 0
        vlm_acc = 0
        while self.step <= self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    # record time
                    self.logger.log('train_time/duration', time.time() - start_time, self.step)
                    start_time = time.time()

                    # logging for frm reward component
                    episode_reward_dict = defaultdict(float)
                    for d in step_frm_dict:
                        for key, value in d.items():
                            episode_reward_dict[key] += value
                    episode_reward_dict = dict(episode_reward_dict)
               
                    # reward function update 
                    if self.step <= self.cfg.FRM_align_max_steps:
                        # rf reward component logging for policy feedback
                        if len(self.FRM_gen.metric_dict) == 0:
                            if "metaworld" in self.cfg.env:
                                self.FRM_gen.metric_dict["socre"] = [round(episode_success, 2)]
                            else:
                                self.FRM_gen.metric_dict["socre"] = [round(true_episode_reward, 2)]
                            for metric in episode_reward_dict.keys():
                                self.FRM_gen.metric_dict[metric] = [round(episode_reward_dict[metric], 2)]
                        else:
                            if "metaworld" in self.cfg.env:
                                self.FRM_gen.metric_dict["socre"].append(round(episode_success, 2))
                            else:
                                self.FRM_gen.metric_dict["socre"].append(round(true_episode_reward, 2))
                            for metric in episode_reward_dict.keys():
                                self.FRM_gen.metric_dict[metric].append(round(episode_reward_dict[metric], 2))

                        if self.step % self.cfg.FRM_align_step == 0:
                            # update reward function
                            eval_episode_freq = max(int(len(self.FRM_gen.metric_dict["socre"]) // 10), 1)
                            rf = self.FRM_gen.reward_func_gen(episode_freq=eval_episode_freq, cur_reward_func=self.FRM)
                            if rf is not None:
                                # if improved, update the reward function
                                self.FRM = rf
                                # relabel
                                if "metaworld" in self.cfg.env:
                                    self.replay_buffer.relabel_with_rf(self.FRM, self.env.target_pos_frm, self.step)
                                elif "softgym_PassWater" in self.cfg.env:
                                    self.replay_buffer.relabel_with_rf(self.FRM, 1.2, self.step)
                                elif "softgym" in self.cfg.env:
                                    self.replay_buffer.relabel_with_rf(self.FRM, None, self.step)
                                else:
                                    raise ValueError
                                print("Relabel with new reward function successful!")
                            else:
                                print("Using the last iter reward function! ")

                    self.logger.log('train_time/query_label', 0, self.step)
                    self.logger.log('train_time/reward_learning', 0, self.step)
                    self.logger.log('train_time/relabel', 0, self.step)

                    self.logger.log('train/reward_learning_acc', reward_learning_acc, self.step)
                    self.logger.log('train/vlm_acc', vlm_acc, self.step)
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps+1), ty='train')
                    if "Cloth" in self.cfg.env:
                        self.logger.log('train/true_reward', episode_end_reward, self.step)
                    
                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/episode_reward_frm', episode_reward_frm, self.step)
                self.logger.log('train/episode_reward_rrm', episode_reward_rrm, self.step)

                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)

                # reset env
                obs = self.env.reset()
                if "metaworld" in self.cfg.env:
                    obs = obs[0]
                # get goal pos
                if "metaworld" in self.cfg.env:
                    self.FRM_gen.target_pos = self.env.target_pos_frm
                elif "Water" in self.cfg.env:
                    self.FRM_gen.target_pos = 1.2

                done = False
                episode_reward, episode_reward_frm, episode_reward_rrm = 0, 0, 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                step_frm_dict = []

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_pre_steps):
                
                self.reward_alpha = self.cfg.reward_alpha
                self.replay_buffer.reward_alpha = self.cfg.reward_alpha
                print(f"set the rm_ alpha: {self.reward_alpha}")
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.RRM.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps) # an average segment reward
                self.RRM.set_teacher_thres_skip(new_margin)
                self.RRM.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                reward_learning_acc, vlm_acc = self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.relabel_with_reward_model()
                
                # reset Q to reward_model + reward_function
                print("Reset ACTOR AND CRITIC")
                self.agent.reset_critic()
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_pre_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.RRM.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.RRM.set_teacher_thres_skip(new_margin)
                        self.RRM.set_teacher_thres_equal(new_margin)
                        
                        # corner case: new total feed > max feed
                        if self.RRM.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.RRM.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        reward_learning_acc, vlm_acc = self.learn_reward()

                        relabel_timer = time.time()
                        # relabel buffer
                        self.relabel_with_reward_model()

                        self.logger.log('train_time/relabel', time.time() - relabel_timer, self.step)
                        print(f"Relabel time: {time.time() - relabel_timer}")

                        interact_count = 0

            if not self.step < self.cfg.num_seed_steps:    
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            try: # for handle stupid gym wrapper change 
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated

            # Get RGB images for each step
            if "metaworld" in self.cfg.env:
                rgb_image = self.env.render()
                rgb_image = rgb_image[::-1, :, :]
                if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                    rgb_image = rgb_image[100:400, 100:400, :]
            elif 'softgym' in self.cfg.env:
                rgb_image = self.env.render(mode='rgb_array', hide_picker=True)
            else:
                rgb_image = self.env.render(mode='rgb_array')

            if 'Water' not in self.cfg.env and 'Rope' not in self.cfg.env:
                rgb_image = cv2.resize(rgb_image, (self.image_height, self.image_width)) # NOTE: resize image here

            # compute reward
            if self.step < (self.cfg.num_seed_steps + self.cfg.num_pre_steps):
                    # Don not use reward model when no first reward learning
                    reward_rrm = 0
            else:
                image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0
                image = image[:, ::self.resize_factor, ::self.resize_factor] # subsample
                image = image.reshape(1, 3, image.shape[1], image.shape[2])
                self.RRM.eval()
                reward_rrm = self.RRM.r_hat(image)
                self.RRM.train()

            # compute reward function reward
            if "metaworld" in self.cfg.env:
                reward_frm, step_frm_info = self.FRM(next_obs, action, self.env.target_pos_frm)
            elif "softgym_PassWater" in self.cfg.env:
                reward_frm, step_frm_info = self.FRM(next_obs, action, 1.2)
            elif "softgym" in self.cfg.env:
                reward_frm, step_frm_info = self.FRM(next_obs, action, None)
            else:
                raise ValueError
            reward_frm = np.clip(reward_frm, -1, 1)
            step_frm_dict.append(step_frm_info)
            reward_frm = reward_frm

            # allow infinite bootstrap
            done = float(done)
            if 'softgym' not in self.cfg.env:
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            else:
                done_no_max = done

            # reward fusion
            reward_frm = (1.0 - self.reward_alpha) * reward_frm
            reward_rrm = self.reward_alpha * reward_rrm

            reward_hat = reward_frm + reward_rrm

            episode_reward += reward_hat
            episode_reward_frm += reward_frm
            episode_reward_rrm += reward_rrm
            
            if "Cloth" in self.cfg.env:
                true_episode_reward = reward
            else:
                true_episode_reward += reward

            if "Cloth" in self.cfg.env and episode_step == 2:
                episode_end_reward = reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.replay_buffer.add(obs, action, reward_hat, reward_frm, reward_rrm, 
                next_obs, done, done_no_max, image=rgb_image[::self.resize_factor, ::self.resize_factor, :])

            if len(self.RRM.all_cached_labels) < self.cfg.max_feedback and self.total_feedback < self.cfg.max_feedback:
                self.RRM.add_data(obs, action, reward, done, img=rgb_image)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            if self.empty_pre_dataset and not self.RRM.preference_data_emptyed and self.total_feedback >= self.cfg.max_feedback:
                self.RRM.empty_preference_data()
            
        self.agent.save(model_save_dir, self.step)
        self.RRM.save(model_save_dir, self.step)
            
@hydra.main(config_path='config/train_CoRe.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
