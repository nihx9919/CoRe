from typing import Optional
from openai import OpenAI
import base64
import os
import re
import time
import warnings
import logging
from FRM.file_process import file_to_string
import json

api_key = os.environ['MY_API_KEY']

# for create a dialogue
class Conversation():
    def __init__(self, system_prompt: Optional[str]) -> None:
        if system_prompt is None:
            self.messages = []
        else:
            self.messages = [
                {
                "role": "system",
                "content": system_prompt
                }
            ]
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0

    def add_system_prompt(self, system_prompt:str) -> None:
        self.messages.append({
            "role": "system",
            "content": system_prompt
        })

    def add_assistant_content(self, assistant_content:str):
        self.messages.append(
            {
            "role": "assistant",
            "content": assistant_content
            }
        )

    def add_user_content(self, contents:list):
        """
        contents format:
        [
            {"type": "text", "data": ""}, 
            {}
            ]
        """
        full_content = []
        for content in contents:
            if content["type"] == "text":
                 full_content.append(
                    {
                    "type": "text", 
                    "text": content["data"]
                    }
                )
            elif content["type"] == "image_url":
                full_content.append(
                    {
                    "type": "image_url",
                    "image_url": 
                        {
                        "url": f"data:image/png;base64,{content['data']}",
                        "detail": "high"
                        }
                    }
                )
            else:
                raise ValueError("content type error. ")
        
        self.messages.append(
            {
            "role": "user",
            "content": full_content
            }
        )
    
    def update_usage(self, usage):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

    def clear_prompt(self):
        # clear prompt without system prompt
        self.messages.clear()



class Agent():
    def __init__(self, system_prompt, model_type="gpt-4.1-mini", temperature=1):
        
        self.client = OpenAI(
            api_key=api_key,
            )
        self.model_type = model_type
        print(f"Using LLM: {model_type}")
        self.conversation = Conversation(system_prompt)
        self.temperature = temperature

    def query(self, sample_num, once_sample_num):
        responses = []
        response_cur = None
        total_samples = 0

        while True:
            if total_samples >= sample_num:
                break
            for attempt in range(20):
                try:
                    response_cur = self.client.chat.completions.create(
                        model=self.model_type,
                        messages=self.conversation.messages,
                        temperature=self.temperature,
                        n=once_sample_num
                    )
                    total_samples += once_sample_num
                    break
                except Exception as e:
                    if attempt >= 10:
                        once_sample_num = max(int(sample_num / 2), 1)
                        print("Current once sample num", once_sample_num)
                    print(f"Failed with error: {e}\n{self.model_type} attempt {attempt + 1}")
                    time.sleep(2)
            if response_cur is None:
                print("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            self.conversation.update_usage(response_cur.usage)

        return responses



