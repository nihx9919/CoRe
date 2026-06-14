#!/usr/bin/env python3
import json
import numpy as np
import re
from FRM.file_process import *
import shutil
from FRM.misc import *
from FRM.code_agent import Agent
import importlib
import logging

task_description_prompts = {
    "metaworld_sweep-into-v2": "to sweep the green cube into the hole with the robot end-effector",
    "metaworld_soccer-v2": "to move the ball into the goal with the robot end-effector",
    "metaworld_drawer-open-v2": "to open the drawer with the robot end-effector", 
    "metaworld_button-press-topdown-v2": "to press the red button down completely from top to bottom with the robot end-effector",
    "metaworld_dial-turn-v2": "to turn the dial with the robot end-effector",
    "metaworld_hammer-v2": "to use the robot end-effector to grasp the hammer and hammer the nail",
    "metaworld_peg-insert-side-v2": "to use the robot end-effector to grasp the peg and insert it into the hole",

    "softgym_RopeFlattenEasy": "to straighten the blue rope",
    "softgym_PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "softgym_ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",
}


class FRMCode:
    def __init__(self, cfg, project_dir, work_dir, reward_model, sample_num=4, once_sample_num=4, llm_model='gpt-4.1-mini'):
        self.cfg = cfg
        self.project_dir = project_dir
        self.work_dir = work_dir
        self.reward_model = reward_model
        if "metaworld" in cfg.env:
            task_mujoco_code_path = f'{project_dir}/metaworld/envs/mujoco/sawyer_xyz/v2/{self.cfg.env.replace("metaworld", "sawyer").replace("-", "_").lower()}.py'
            if "sweep" in self.cfg.env:
                task_mujoco_code_path = f'{project_dir}/metaworld/envs/mujoco/sawyer_xyz/v2/sawyer_sweep_into_goal_v2.py'
            elif "insert" in self.cfg.env:
                task_mujoco_code_path = f'{project_dir}/metaworld/envs/mujoco/sawyer_xyz/v2/sawyer_peg_insertion_side_v2.py'
            
            with open(task_mujoco_code_path, "r") as task_file:
                lines = task_file.readlines()
                add_line = "self.target_pos_frm"
                target_line = "return self._get_obs()"
                for i, line in enumerate(lines):
                    if add_line in line:
                        break
                    if target_line in line:
                        indent = " " * (len(line) - len(line.lstrip()))
                        lines.insert(i, indent + "self.target_pos_frm = self._target_pos" + "\n")
                        break
            with open(task_mujoco_code_path, "w") as taks_file:
                taks_file.writelines(lines)

        # initialize FRM
        self.load_prompt()
        self.frm_gen_agent = Agent(self.initial_system, model_type=llm_model)
        self.frm_gen_agent.conversation.add_user_content([{"type": "text", "data": self.initial_user}])

        self.iter = 0
        self.sample_num = sample_num
        self.once_sample_num = once_sample_num

        self.best_frm_response = None
        
        # for policy feedback 
        # record metric every episode
        self.metric_dict = {}

        self.best_alignment_percent = None
        self.target_pos = None

    
    def load_prompt(self):

        task_info_file = f"{self.project_dir}/envs/env_info/{self.cfg.env.lower()}_info.py"
        shutil.copy(task_info_file, f"env_info.py")
        task_info_code_string = file_to_string(task_info_file)

        task_description = task_description_prompts[self.cfg.env]

        prompt_dir = f'{self.project_dir}/FRM/prompts'
        self.initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
        self.code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
        self.code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
        self.initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
        reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
        self.policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
        self.execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

        self.initial_system = self.initial_system.format(task_reward_signature_string=reward_signature) + self.code_output_tip
        self.initial_user = self.initial_user.format(task_info_code_string=task_info_code_string, task_description=task_description)

    def dynamic_import_frm(self, FRM_file):
        # dynamic import FRM
        try:
            FRM_py = importlib.import_module(f"{FRM_file}")
            logging.info(f"Using FRM file: {FRM_file}")
            self.iter += 1
            self.metric_dict = {}
            return getattr(FRM_py, "reward_function")
        except Exception as e:
            raise ImportError(f"Import FRM for initialization fail! {e}")
            
    def FRM_generate(self, max_tries = 3, episode_freq=0, cur_frm=None):

        if self.iter > 0:
            # prepare for FRM check
            # construct policy feedback
            policy_feedback = self.construct_policy_feedback(episode_freq)
            # update llm agent conversation message
            self.update_message(policy_feedback)
            # evaluate current frm
            s_1, a_1, s_2, a_2, label_gt = self.set_preference_data()
            self.cur_alignment_score = self.frm_reward_preference_alignment(cur_frm, s_1, a_1, s_2, a_2, label_gt)
            logging.info(f"FRM Check Start, Cur Alignment Socre: {self.cur_alignment_score}")

        for i in range(max_tries):
            logging.info(f"Iteration {self.iter}, Attempt {i}, Sample Num {self.sample_num}, Once Sample Num {self.once_sample_num}")
            # initialize frm
            if self.iter == 0:
                best_frm_id = self.gen_frm_by_train(self.iter)
            # refine frm
            elif self.iter > 0:
                assert episode_freq != 0
                assert cur_frm != None
                best_frm_id = self.update_frm_by_preference(s_1, a_1, s_2, a_2, label_gt)

            if best_frm_id is not None:
                best_FRM_name = f"env_iter{self.iter}_response{best_frm_id}"
                logging.info(f"best FRM: {best_FRM_name}")
                frm = self.dynamic_import_frm(f"{best_FRM_name}")
                return frm
            else:
                continue

        return None
    
    def get_frm_response(self, iter_id):

        # query LLM and output FRM
        responses = self.frm_gen_agent.query(self.sample_num, self.once_sample_num)

        # Save dictionary as JSON file
        with open(f"iter{iter_id}_messages.json", 'w') as file:
            json.dump(self.frm_gen_agent.conversation.messages, file, indent=4)

        for response_id in range(self.sample_num):
            response_cur = responses[response_id].message.content
            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])

            with open(f"env_iter{iter_id}_response{response_id}.py", 'w', encoding='utf-8') as file:
                file.writelines("import numpy as np" + '\n')
                file.write(code_string + '\n')
        
        return responses   

    def gen_frm_by_train(self, iter_id):
        assert iter_id == 0
        DUMMY_FAILURE = -100000
        responses = self.get_frm_response(self.iter)

        # set_freest_gpu()
        rl_runs = []
        frm_init_steps = self.cfg.num_train_steps // 10
        # frm_init_steps = 50000
        agent_yml = "sac"
        if "Cloth" in self.cfg.env:
            agent_yml = "sac_cloth"

        for response_id in range(self.sample_num):
            rl_filepath = f"env_iter{iter_id}_response{response_id}.txt"

            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{self.project_dir}/FRM/FRM_init.py',
                                            'hydra/output=frm',
                                            f'agent={agent_yml}',
                                            f'env={self.cfg.env}', f'num_train_steps={frm_init_steps}', f'num_seed_steps={self.cfg.num_seed_steps}',
                                            f'eval_frequency={frm_init_steps}', 'num_eval_episodes=10',
                                            f'iter={iter_id}', f'response={response_id}',
                                            f'work_dir_main={self.work_dir}', 
                                            f'save_model_step={frm_init_steps}',
                                            ], stdout=f, stderr=f)
            block_until_training(rl_filepath, log_status=True, iter_num=self.iter, response_id=response_id)  # return when success or error
            rl_runs.append(process)

        contents = []
        scores = []
        reward_correlations = []
        
        exec_success = False
        for response_id, rl_run in enumerate(rl_runs):
            rl_run.communicate()
            rl_filepath = f"env_iter{iter_id}_response{response_id}.txt"
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()
            except:
                logging.info("Code Run cannot be executed due to function signature error!")
                scores.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break
                tensorboard_logdir = line.split(':')[-1].strip()
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_episodes = np.array(tensorboard_logs['train/episode']).shape[0]
                episode_freq = max(int(max_episodes // 10), 1)

                content += self.policy_feedback.format(episode_freq=episode_freq)

                # Compute Correlation between Human-Engineered and FRM
                if "train/true_episode_reward" in tensorboard_logs and "train/episode_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["train/true_episode_reward"])
                    frm_reward = np.array(tensorboard_logs["train/episode_reward"])
                    reward_correlation = np.corrcoef(gt_reward, frm_reward)[0, 1]  # 2*2 matrix
                    reward_correlations.append(reward_correlation)

                for metric in tensorboard_logs:
                    if "metaworld" in self.cfg.env:
                        if "frm" in metric or "train/episode_success" in metric:
                            metric_cur = ['{:.2f}'.format(x) for x in
                                          tensorboard_logs[metric][::episode_freq]]  # record every episode_freq
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            metric_cur_min = min(tensorboard_logs[metric])
                            if "train/episode_success" == metric:
                                scores.append(metric_cur_mean)
                                content += f"task score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                            else:
                                metric_name = metric.split("frm_")[1]
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                    elif "softgym" in self.cfg.env:
                        if "frm" in metric or "train/true_episode_reward" in metric:
                            metric_cur = ['{:.2f}'.format(x) for x in
                                          tensorboard_logs[metric][::episode_freq]]  # record every episode_freq
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            metric_cur_min = min(tensorboard_logs[metric])
                            if "train/true_episode_reward" == metric:
                                scores.append(metric_cur_mean)
                                content += f"task score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                            else:
                                metric_name = metric.split("frm_")[1]
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
            else:
                # Otherwise, provide execution traceback error feedback
                scores.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)

            contents.append(content)

        if not exec_success:
            logging.info("All code generation failed!")
            return None

        # Select the best code sample based on the task score
        logging.info(f"Scores :{scores}")
        best_sample_idx = np.argmax(np.array(scores))
        best_content = contents[best_sample_idx]
        max_score = scores[best_sample_idx]
        max_score_reward_correlation = reward_correlations[best_sample_idx]

        self.best_frm_response = responses[best_sample_idx].message.content
        
        logging.info(f"Iteration {self.iter}: Best Generation ID: {best_sample_idx} Max Score: {max_score} Max Score Reward Correlation: {max_score_reward_correlation}")
        logging.info(f"Iteration {self.iter}: Output Content:\n" + responses[best_sample_idx].message.content + "\n")
        logging.info(f"Iteration {self.iter}: User Content:\n" + best_content + "\n")

        return best_sample_idx
    
    def construct_policy_feedback(self, episode_freq):
        
        content = ''
        content += self.policy_feedback.format(episode_freq=episode_freq)

        # Add reward components log to the feedback
        for metric in self.metric_dict.keys():
            metric_cur = self.metric_dict[metric][::episode_freq]
            metric_cur_max = max(self.metric_dict[metric])
            metric_cur_mean = sum(self.metric_dict[metric]) / len(self.metric_dict[metric])
            metric_cur_min = min(self.metric_dict[metric])
            content += f"{metric}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
        logging.info(content)
        content += self.code_feedback
        content += self.code_output_tip
        return content
    
    def update_message(self, policy_feedback):
        cur_message = self.frm_gen_agent.conversation.messages
        if len(cur_message) == 4:
            del self.frm_gen_agent.conversation.messages[2:4]
        self.frm_gen_agent.conversation.add_assistant_content(self.best_frm_response)
        self.frm_gen_agent.conversation.add_user_content([{"type": "text", "data": policy_feedback}])
        
    def update_frm_by_preference(self, s_1, a_1, s_2, a_2, label_gt):

        # generate new FRMs
        responses = self.get_frm_response(self.iter)
        best_frm_id = -1
        for frm_id in range(self.sample_num):
            FRM_file_name = f"env_iter{self.iter}_response{frm_id}"
            try:
                frm = getattr(importlib.import_module(FRM_file_name), "reward_function")
                alignment_percent = self.frm_reward_preference_alignment(frm, s_1, a_1, s_2, a_2, label_gt)
                logging.info(f"Checking FRM: {FRM_file_name}! checking/cur alignment: {alignment_percent:.4f}/{self.cur_alignment_score:.4f}")
                if alignment_percent > self.cur_alignment_score:
                    best_frm_id = frm_id
                    self.cur_alignment_score = alignment_percent
                    logging.info(f"FRM Check Success, Cur Alignment Score: {alignment_percent}\n")

            except Exception as e:
                logging.info(f"FRM Check {FRM_file_name} Failed! {e}")
                continue
        
        if best_frm_id >=0:
            # get improved ! update current best FRM
            self.best_frm_response = responses[best_frm_id].message.content
            return best_frm_id
        else:
            return None
        

    def frm_reward_preference_alignment(self, rf, s1, a1, s2, a2, label_gt):
        """
        return the correct labels labeled by frm
        """
        r1 = []
        r2 = []
        for seg in range(a1.shape[0]):
            r1_seg = 0
            r2_seg = 0
            for t in range(a1.shape[1]):
                r1_seg_t, _ = rf(s1[seg, t], a1[seg, t], self.target_pos)
                r2_seg_t, _ = rf(s2[seg, t], a2[seg, t], self.target_pos)
                r1_seg += r1_seg_t
                r2_seg += r2_seg_t
            r1.append(r1_seg)
            r2.append(r2_seg)
        label = np.array(r1) < np.array(r2)
        try:
            acc = sum(np.array(label_gt) == label.astype(float)) / len(label_gt)
        except:
            print("No labels for checking!")
            acc = 0
        return acc

    def set_preference_data(self):
        label_gt = self.reward_model.buffer_label[:self.reward_model.buffer_index, :].squeeze()
        s_1 = self.reward_model.buffer_rf1[:self.reward_model.buffer_index, :, :self.reward_model.ds]
        a_1 = self.reward_model.buffer_rf1[:self.reward_model.buffer_index, :, self.reward_model.ds:]
        s_2 = self.reward_model.buffer_rf2[:self.reward_model.buffer_index, :, :self.reward_model.ds]
        a_2 = self.reward_model.buffer_rf2[:self.reward_model.buffer_index, :, self.reward_model.ds:]

        return s_1, a_1, s_2, a_2, label_gt