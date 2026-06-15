# for sweep
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_sweep-into-v2 \
        seed=$seed \
        exp_name=core \
        segment=20 \
        RRM_batch=20 \
        max_feedback=500 \
        num_train_steps=500000 \
        num_eval_episodes=10 \
        num_pre_steps=19000 \
        save_eval_video=False \
        FRM_align_step=25000 \
        FRM_align_max_steps=100000 \
        use_FRM_online=False
done

# for soccer
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_soccer-v2 \
        seed=$seed \
        exp_name=core \
        segment=20 \
        RRM_batch=20 \
        max_feedback=500 \
        num_train_steps=500000 \
        num_eval_episodes=10 \
        num_pre_steps=19000 \
        save_eval_video=False \
        FRM_align_step=25000 \
        FRM_align_max_steps=100000 \
        use_FRM_online=False
done

# for drawer open
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_drawer-open-v2 \
        seed=$seed \
        exp_name=core \
        segment=20 \
        RRM_batch=20 \
        max_feedback=500 \
        num_train_steps=500000 \
        num_eval_episodes=10 \
        num_pre_steps=19000 \
        save_eval_video=False \
        FRM_align_step=25000 \
        FRM_align_max_steps=100000 \
        use_FRM_online=False
done

# botton pres topdown
for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_button-press-topdown-v2 \
        seed=$seed \
        exp_name=core \
        segment=20 \
        RRM_batch=20 \
        max_feedback=500 \
        num_train_steps=500000 \
        num_eval_episodes=10 \
        num_pre_steps=19000 \
        save_eval_video=False \
        FRM_align_step=25000 \
        FRM_align_max_steps=100000 \
        use_FRM_online=False
done

for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_dial-turn-v2 \
        seed=$seed \
        exp_name=core \
        segment=20 \
        RRM_batch=20 \
        max_feedback=500 \
        num_train_steps=500000 \
        num_eval_episodes=10 \
        num_pre_steps=19000 \
        save_eval_video=False \
        FRM_align_step=25000 \
        FRM_align_max_steps=100000 \
        use_FRM_online=False
done

for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_hammer-v2 \
        seed=$seed \
        exp_name=core \
        segment=20 \
        RRM_batch=20 \
        max_feedback=500 \
        num_train_steps=500000 \
        num_eval_episodes=10 \
        num_pre_steps=19000 \
        save_eval_video=False \
        FRM_align_step=25000 \
        FRM_align_max_steps=100000 \
        use_FRM_online=False
done

for seed in 0 1 2; do
    python train_CoRe.py \
        env=metaworld_peg-insert-side-v2 \
        seed=$seed \
        exp_name=core \
        segment=20 \
        RRM_batch=20 \
        max_feedback=500 \
        num_train_steps=500000 \
        num_eval_episodes=10 \
        num_pre_steps=19000 \
        save_eval_video=False \
        FRM_align_step=25000 \
        FRM_align_max_steps=100000 \
        use_FRM_online=False
done
