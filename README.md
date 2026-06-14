# CoRe
This is the official codebase for:  
[CoRe: Combined Rewards with Vision-Language Model Feedback for Preference-Aligned Reinforcement Learning](https://core-2026.github.io/),  
Ni Hexian, Tao Lu, Yinghao Cai,  
ICML 2026.

## Install
Install the conda environment via:
```
conda env create -f conda_env.yaml
conda activate core
```

For GPU support, install the CUDA build of PyTorch after activating the environment (example for CUDA 11.7):
```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```


For SoftGym tasks (cloth fold, straighten rope, pass water, etc.), please refer to https://github.com/Xingyu-Lin/softgym?tab=readme-ov-file for compiling SoftGym.

## API Key and Model Configuration
CoRe uses a VLM for preference labeling (RRM) and an LLM for FRM code generation. Before running experiments:

1. Set the API key environment variable:
```
export MY_API_KEY=your_api_key
```

2. Configure the model and API endpoint in the source code:
- **RRM (preference labeling)**: edit `model_type` and `base_url` in `RRM/preference_label.py` (default: `gemini-2.5-flash-lite`).
- **FRM (reward code generation)**: edit `base_url` in `FRM/code_agent.py`, and `llm_model` in `FRM/reward_code.py` (default: `gpt-4.1-mini`).

Both modules read the API key from the `MY_API_KEY` environment variable and use an OpenAI-compatible API interface.

## Run Experiments
After environment setup and API configuration, run the provided shell scripts:

- **MetaWorld tasks** (sweep, soccer, drawer open, button press, dial turn, hammer, peg insert):
```
bash run_meta.sh
```

- **SoftGym tasks** (cloth fold, rope flatten, pass water):
```
bash run-soft.sh
```

## Acknowledgements
- We thank the authors of [RL-VLM-F](https://github.com/yufeiwang63/RL-VLM-F) for open sourcing their code, which our codebase is built upon.
- We thank the authors of [Eureka](https://github.com/eureka-research/eureka) for their work on LLM-based reward design, which inspired our FRM module.

## Citation
If you find this codebase / paper useful in your research, please consider citing:
```
@InProceedings{ni2026core,
  title     = {CoRe: Combined Rewards with Vision-Language Model Feedback for Preference-Aligned Reinforcement Learning},
  author    = {Ni, Hexian and Lu, Tao and Cai, Yinghao},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
