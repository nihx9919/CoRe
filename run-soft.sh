for seed in 0 1 2; do
python train_PEBBLE_rf.py \
    env=softgym_ClothFoldDiagonal agent=sac_cloth \
    seed=$seed \
    exp_name=vlm_video_tanh \
    segment=3 \
    num_train_steps=15000 num_pre_steps=250 num_interact=1000 num_seed_steps=250 save_model_step=2500 \
    max_feedback=150 reward_batch=10 reward_update=25 resnet=1 kl_weight=5 reward_type=gemini_preference_video \
    eval_frequency=250 num_eval_episodes=3 save_eval_video=False \
    num_rf_fre=2400 num_rf_max_steps=9600 use_rf_online=False
done

for seed in 0 1 2 ; do
python train_PEBBLE_rf.py \
    env=softgym_RopeFlattenEasy \
    seed=$seed \
    exp_name=vlm_video_tanh \
    segment=3 \
    num_train_steps=101000 num_pre_steps=9000 num_interact=5000 save_model_step=25000 \
    max_feedback=200 reward_batch=10 reward_update=30 resnet=1 reward_lr=1e-4 kl_weight=5 reward_type=gemini_preference_video \
    num_eval_episodes=3 save_eval_video=False \
    num_rf_fre=25000 num_rf_max_steps=100000 use_rf_online=False
done
wait

for seed in 0 1 2; do
python train_PEBBLE_rf.py \
    env=softgym_PassWater \
    seed=$seed \
    exp_name=vlm_video \
    segment=3 \
    num_train_steps=100050 num_pre_steps=9000 num_interact=5000 save_model_step=25000 \
    max_feedback=200 reward_batch=10 reward_update=30 resnet=1 reward_lr=1e-4 kl_weight=5 reward_type=gemini_preference_video \
    num_eval_episodes=3 save_eval_video=False \
    num_rf_fre=24975 num_rf_max_steps=100000 use_rf_online=False
done
wait


