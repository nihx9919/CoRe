from openai import OpenAI
import base64
import pickle as pkl
import time
from RRM.prompt import (analyse_video_prompt, task_init_prompt, task_done_prompt, add_video_prompt, 
                         video_question_prompt, 
                         get_label_prompt, get_label_system_prompt)
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
from FRM.code_agent import Agent
import os
import re
from scipy.interpolate import interp1d
from RRM.prompt import env_goal_prompts

model_type = "gemini-2.5-flash-lite"
api_key = os.environ['MY_API_KEY']

def get_n_indices(length, k):
    return [int(round(i)) for i in np.linspace(0, length - 1, k)]

def np_to_base64_image(np_img: np.ndarray, format: str = "png") -> str:
    img = Image.fromarray(np_img)
    buffered = BytesIO()
    img.save(buffered, format=format.upper())
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def create_render_video(frames_a, frames_b, sampled_a, sampled_b, idx, label, weight_a, weight_b, resize=(300, 300)):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    sep_vieo_width = 20
    sep_frame_width = 5
    max_width = 2 * resize[1] + sep_vieo_width

    def resize_and_cvtcolor_frames(frames, size):
        return [cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_RGB2BGR) for f in frames]

    # Resize upper dynamic video frames
    frames_a = resize_and_cvtcolor_frames(frames_a, resize)
    frames_b = resize_and_cvtcolor_frames(frames_b, resize)
    num_frames = len(frames_a)

    # Resize sampled frames based on resized video height
    sample_size = int((max_width - (len(sampled_a) - 1) * sep_frame_width) / len(sampled_a))
    sample_size = (sample_size, sample_size)
    sampled_a = resize_and_cvtcolor_frames(sampled_a, sample_size)
    sampled_b = resize_and_cvtcolor_frames(sampled_b, sample_size)

    # Create a vertical separator
    def assemble_sample_strip(sampled_frames, label, weights=None):
        sep = np.ones((sampled_frames[0].shape[0], sep_frame_width, 3), dtype=np.uint8) * 255
        spaced_frames = []

        for i, f in enumerate(sampled_frames):
            frame_with_text = f.copy()

            if weights is not None and i < len(weights):
                text = f"{weights[i]:.2f}"
                cv2.putText(frame_with_text, text, (150, 15), font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)

            spaced_frames.append(frame_with_text)

            if i < len(sampled_frames) - 1:
                spaced_frames.append(sep)

        strip = cv2.hconcat(spaced_frames)

        h = 30
        text_img = np.ones((h, strip.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(text_img, label, (10, h - 10), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return cv2.vconcat([text_img, strip])

    strip_a = assemble_sample_strip(sampled_a, "Sampled Frames from Video 0", weight_a)
    strip_b = assemble_sample_strip(sampled_b, "Sampled Frames from Video 1", weight_b)

    # Create ID + Label info strip
    info_height = 60
    info_img = np.ones((info_height, max_width, 3), dtype=np.uint8) * 255
    y = 25
    for line in [f"ID: {idx}", f"Label: {label}"]:
        cv2.putText(info_img, line, (10, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += int(30 * font_scale) + 5

    if strip_a.shape[1] < info_img.shape[1]:
        strip_a = cv2.hconcat([strip_a, np.ones((strip_a.shape[0], info_img.shape[1] - strip_a.shape[1], 3), dtype=np.uint8) * 255])
        strip_b = cv2.hconcat([strip_b, np.ones((strip_a.shape[0], info_img.shape[1] - strip_b.shape[1], 3), dtype=np.uint8) * 255])
    # Create video frames
    result_frames = []
    for t in range(num_frames):
        sep = np.ones((resize[0], sep_vieo_width, 3), dtype=np.uint8) * 255
        top = cv2.hconcat([frames_a[t], sep, frames_b[t]])
        
        full = cv2.vconcat([top, info_img, strip_a, strip_b])
        result_frames.append(full)

    return result_frames


def save_render_videos(recorded_frames, output_dir, time_string, index, label):

    h, w, _ = recorded_frames[0].shape
    save_path = os.path.join(f'{output_dir}', f"ID_{index}_L_{label}_{time_string}.mp4")
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    for frame in recorded_frames:
        writer.write(frame)
    writer.release()


class PreAgent(Agent):

    def __init__(self, env_name, segment_size, use_cache=False, project_dir='./'):
        super().__init__(None, model_type, temperature=0)

        self.project_dir = project_dir
        self.env_name = env_name
        self.segment_size = segment_size
        self.task_description = env_goal_prompts[env_name]
        self.init_rgb_image = np_to_base64_image(self.load_init_image())
        self.goal_rgb_image = np_to_base64_image(self.load_goal_image())

    def recreate_client(self):
        return OpenAI(
            api_key=api_key,
        )
    
    def load_goal_image(self):
        print(f'{self.project_dir}/goal_image/{self.env_name}-done.jpg')
        image_bgr = cv2.imread(f'{self.project_dir}/goal_image/{self.env_name}-done.jpg')
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    def load_init_image(self):
        print(f'{self.project_dir}/goal_image/{self.env_name}-init.jpg')
        image_bgr = cv2.imread(f'{self.project_dir}/goal_image/{self.env_name}-init.jpg')
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def fill_extract_video_prompts(self, img_t_1, img_t_2):
        self.conversation.clear_prompt()

        self.conversation.add_user_content([{"type": "text", "data": analyse_video_prompt}])
        # add task description prompt
        self.conversation.add_user_content([{"type": "text", "data": task_init_prompt.format(self.task_description)}, 
                                            {"type": "image_url", "data": self.init_rgb_image}, 
                                            {"type": "text", "data": task_done_prompt}, 
                                            {"type": "image_url", "data": self.goal_rgb_image}, 
                                            ])
        self.conversation.add_user_content([{"type": "text", "data": add_video_prompt}])
        # add video clip A
        self.conversation.add_user_content([{"type": "text", "data": "**Start of Video clip A**"}])
        for idx in range(len(img_t_1)):
            self.conversation.add_user_content([{"type": "text", "data": f"Frame {idx+1} of Video clip A"}, 
                                                {"type": "image_url", "data": np_to_base64_image(img_t_1[idx])}])
        self.conversation.add_user_content([{"type": "text", "data": "**End of Video clip A**"}])

        # add video clip B
        self.conversation.add_user_content([{"type": "text", "data": "**Start of Video clip B**"}])
        for idx in range(len(img_t_2)):
            self.conversation.add_user_content([{"type": "text", "data": f"Frame {idx+1} of Video clip B"}, 
                                                {"type": "image_url", "data": np_to_base64_image(img_t_2[idx])}])
        self.conversation.add_user_content([{"type": "text", "data": "**End of Video clip B**"}])

        # add question prompt
        self.conversation.add_user_content([{"type": "text", "data": video_question_prompt}, 
                                            ])
    def fill_get_label_prompts(self, video_info):
        self.conversation.clear_prompt()
        self.conversation.add_system_prompt(get_label_system_prompt)
        self.conversation.add_user_content([{"type": "text", "data": get_label_prompt.format(self.task_description, video_info)}])

    def interpolate_to_weight_dist(self, weights, sample_indices):

        sample_indices = np.array(sample_indices)
        interp_func = interp1d(sample_indices, weights, kind='linear', bounds_error=False, fill_value=(weights[0], weights[-1]))
        weight_dist = interp_func(np.arange(self.segment_size))
        weight_dist = np.clip(weight_dist, 1e-6, None)
        return weight_dist

    def get_important_weight(self, s):
        matches = re.findall(r'([A-Z])\s*:\s*\[\s*([^\]]+?)\s*\]', s, re.DOTALL)

        if len(matches) != 2 or set(k for k, _ in matches) != {'A', 'B'}:
            raise ValueError("Get important weight error!")

        return {
            k: np.fromstring(v.replace('\n', '').strip(), sep=',')
            for k, v in matches
            }

    def extract_answer(self, response):

        decision_match = re.search(r"#decision#:\s*(.+)", response)
        decision = decision_match.group(1).strip() if decision_match else None

        important_weight = self.get_important_weight(response)
        weight_a, weight_b = important_weight['A'], important_weight['B']

        if "-1" in decision:
            return -1, None, None
        elif "A" in decision:
            return 0, weight_a, weight_b
        elif "B" in decision:
            return 1, weight_a, weight_b
        else:
            raise ValueError("Extract answer error!")

    # video1, video2, time_string, img_save_path, idx
    def video_preference_label(self, img_t_1, img_t_2, time_string="test", idx=0):
        output_dir = f"./vlm_output/{time_string}"
        output_video_dir = os.path.join(output_dir, "video")
        os.makedirs(output_video_dir, exist_ok=True)

        max_retries = 7
        success = False
        try_cnt = 0
        query_image_num = 3
        sample_indices = get_n_indices(img_t_1.shape[0], query_image_num)
        query_img_t_1 = img_t_1[sample_indices]
        query_img_t_2 = img_t_2[sample_indices]

        while not success and try_cnt < max_retries:
            if self.client == None:
                self.client = self.recreate_client()
            try:
                self.fill_extract_video_prompts(query_img_t_1, query_img_t_2)
                video_info = self.query(sample_num=1, once_sample_num=1)[0].message.content
                self.fill_get_label_prompts(video_info)
                print(video_info)
                answer = self.query(sample_num=1, once_sample_num=1)[0].message.content
                print(answer)
                label, weight_a, weight_b = self.extract_answer(answer)
                with open(f"{output_video_dir}/ID_{str(idx)}_L_{label}_{time_string}" + ".txt", "w") as f:
                    f.write(video_info)
                    f.write("\n\n")
                    f.write(answer)
                print(f"{model_type}: ### label: {label} ###\n")
                success = True
                
            except Exception as e:
                print(f"failed. Error: {str(e)} \n \n{model_type} attempt {try_cnt + 1}")
                time.sleep(1)
                try_cnt += 1
                
                if try_cnt % 3 == 0:
                    print(f"Recreate client!")
                    self.client = None

                if try_cnt >= max_retries:
                    label = -1  # -1 indicates query failure
            
        if label == -1 and not success:
            print(f"Query completely failed after {max_retries} retries")
        elif label == -1 and success:
            print(f"### Unuseful label:{label} ###")

        render_frames = create_render_video(img_t_1, img_t_2, query_img_t_1, query_img_t_2, idx, label, weight_a, weight_b, resize=(300, 300))
        save_render_videos(render_frames, output_video_dir, time_string, idx, label)

        if label != -1:
            try:
                weight_dist_a = self.interpolate_to_weight_dist(weight_a, sample_indices)
                weight_dist_b = self.interpolate_to_weight_dist(weight_b, sample_indices)
                return label, weight_dist_a, weight_dist_b
            except:
                return -1, None, None
        else:
            return -1, None, None
