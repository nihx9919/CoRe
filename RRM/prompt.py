import copy

env_goal_prompts = {
    "metaworld_sweep-into-v2": "to minimize the distance between the green cube and the hole",
    "metaworld_drawer-open-v2": "to open the drawer", 
    "metaworld_soccer-v2": "to move the soccer ball into the goal", 
    "metaworld_dial-turn-v2": "to turn the red line to the bottom of the dial",
    "metaworld_button-press-topdown-v2": "to press the red button down completely from top to bottom",
    "metaworld_hammer-v2": "to hammer the grey nail completely in with a red hammer",
    "metaworld_peg-insert-side-v2": "to insert the green peg into the hole of the red bolck",

    "softgym_RopeFlattenEasy": "to straighten the blue rope",
    "softgym_PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "softgym_ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",
}

analyse_video_prompt = """
You will be given a task, and images showing the task at 0% (initial) and 100% (completed).
"""
# describe the task, including the textual task, task initialization image and the task done image. 
task_init_prompt = """
The task is {}. The task initialization(the task has not been completed at all) image is as follows:
"""
task_done_prompt = """
The task completion(the task has been completely completed) image is as follows:
"""
add_video_prompt = """
You will now analyze two video clips (A and B) consisting of multiple frames each. 
Important evaluation rules:
1. A frame that moves away from the initial state but toward the goal state should be considered progress.
2. Do NOT treat frames that differ from the initialization image as worse by default. Always evaluate based on proximity to the task completion image.
3. The task progress score must be between 0.0 (same as initialization) and 1.0 (same as completion). Use intermediate scores like 0.3, 0.7, etc. Negative progress is not allowed.
4. Only describe objects that are directly involved in the task. Ignore irrelevant background elements or motion that is unrelated to task completion.
"""

video_question_prompt = """
Now answer the following:
(1) For Video clip A: What are the differences between each frame of video clip A and the task initialization image and the task completion image?
* Frame 1: 
  - Change since task initialization: [...]
  - What remains to task completion: [...]
  - Evaluate task progress: [a value between 0.0 and 1.0]
* Frame 2: ...
* Frame 3: ...

(2) For Video clip B: What are the differences between each frame of video clip B and the task initialization image and the task completion image?
* Frame 1: 
  - Change since task initialization: [...]
  - What remains to task completion: [...]
  - Evaluate task progress: [a value between 0.0 and 1.0]
* Frame 2: ...
* Frame 3: ...

(3) For both Video clips: Compare the task completion progress in each frame of video A and B:
* Frame 1: #Which is closer to the completion image, and why#
* Frame 2: ...
* Frame 3: ...
"""

get_label_system_prompt = """
You are a helpful assistant. 
"""

get_label_prompt = """
You are comparing two video clips A and B, showing partial execution of the following task:
Task: {}

You are given the two video clips A and B analysis:
{}

Your goal is to determine **which video better completes the task** and estimate how close each frame is to the completion state.

Please strictly follow the rules:
1. **The video that is closer to the task completion image in its final frame is preferred**.
2. Do NOT prefer a video just because it starts from the initialization image.
3. Ignore whether the video starts correctly - we only care how far it progresses.
4. Completion scores range from 0.0 (looks like init) to 1.0 (looks like completion). Use intermediate scores if necessary.
If both videos are similarly incomplete, you may return -1.

Format your output exactly like this:
#why#: [Explain clearly which video is closer to completion and why]
#decision#: A or B or -1
#evaluate task completion for each frame in the video clip#:  
A: [score1, score2, score3]  
B: [score1, score2, score3]
"""