# for sweep
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_sweep-into-v2 \
        seed=$seed \
        exp_name=project_demo \
        segment=20 \
        reward_batch=20 \
        max_feedback=500 \
        kl_weight=5 \
        num_train_steps=500000 \
        num_eval_episodes=3 \
        reward_type=gemini_preference_video \
        num_pre_steps=19000 \
        save_eval_video=False \
        num_rf_fre=25000 \
        num_rf_max_steps=100000 \
        use_rf_online=False
done

# for soccer
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_soccer-v2 \
        seed=$seed \
        exp_name=vlm_video_0_5 \
        segment=20 \
        reward_batch=20 \
        max_feedback=500 \
        kl_weight=5 \
        num_train_steps=500000 \
        num_eval_episodes=3 \
        reward_type=gemini_preference_video \
        num_pre_steps=19000 \
        save_eval_video=False \
        num_rf_fre=25000 \
        num_rf_max_steps=100000 \
        use_rf_online=False
done

# for drawer open
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_drawer-open-v2 \
        seed=$seed \
        exp_name=vlm_video_0_5 \
        segment=20 \
        reward_batch=20 \
        max_feedback=500 \
        kl_weight=5 \
        num_train_steps=500000 \
        num_eval_episodes=3 \
        reward_type=gemini_preference_video \
        num_pre_steps=19000 \
        save_eval_video=False \
        num_rf_fre=25000 \
        num_rf_max_steps=100000 \
        use_rf_online=False
done

# botton pres topdown
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_button-press-topdown-v2 \
        seed=$seed \
        exp_name=vlm_video_0_5 \
        segment=20 \
        reward_batch=20 \
        max_feedback=500 \
        kl_weight=5 \
        num_train_steps=500000 \
        num_eval_episodes=3 \
        reward_type=gemini_preference_video \
        num_pre_steps=19000 \
        save_eval_video=False \
        num_rf_fre=25000 \
        num_rf_max_steps=100000 \
        use_rf_online=False
done

for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_dial-turn-v2 \
        seed=$seed \
        exp_name=vlm_video_0_5 \
        segment=20 \
        reward_batch=20 \
        max_feedback=500 \
        kl_weight=5 \
        num_train_steps=500000 \
        num_eval_episodes=3 \
        reward_type=gemini_preference_video \
        num_pre_steps=19000 \
        save_eval_video=False \
        num_rf_fre=25000 \
        num_rf_max_steps=100000 \
        use_rf_online=False
done

for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_hammer-v2 \
        seed=$seed \
        exp_name=vlm_video_0_5 \
        segment=20 \
        reward_batch=20 \
        max_feedback=500 \
        kl_weight=5 \
        num_train_steps=500000 \
        num_eval_episodes=3 \
        reward_type=gemini_preference_video \
        num_pre_steps=19000 \
        save_eval_video=False \
        num_rf_fre=25000 \
        num_rf_max_steps=100000 \
        use_rf_online=False
done

for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_peg-insert-side-v2 \
        seed=$seed \
        exp_name=vlm_video_0_5 \
        segment=20 \
        reward_batch=20 \
        max_feedback=500 \
        kl_weight=5 \
        num_train_steps=500000 \
        num_eval_episodes=3 \
        reward_type=gemini_preference_video \
        num_pre_steps=19000 \
        save_eval_video=False \
        num_rf_fre=25000 \
        num_rf_max_steps=100000 \
        use_rf_online=False
done
