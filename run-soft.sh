for seed in 0 1 2; do
python train_CoRe.py \
    env=softgym_ClothFoldDiagonal agent=sac_cloth \
    seed=$seed \
    experiment=core \
    segment=3 \
    num_train_steps=15000 num_pre_steps=250 num_interact=1000 num_seed_steps=250 save_model_step=2500 \
    max_feedback=150 RRM_batch=10 RRM_update=25 resnet=1 \
    eval_frequency=250 num_eval_episodes=10 save_eval_video=False \
    FRM_align_step=2400 FRM_align_max_steps=9600 use_FRM_online=False
done

for seed in 0 1 2 ; do
python train_CoRe.py \
    env=softgym_RopeFlattenEasy \
    seed=$seed \
    experiment=core \
    segment=3 \
    num_train_steps=101000 num_pre_steps=9000 num_interact=5000 save_model_step=25000 \
    max_feedback=200 RRM_batch=10 RRM_update=30 resnet=1 RRM_lr=1e-4 \
    num_eval_episodes=10 save_eval_video=False \
    FRM_align_step=25000 FRM_align_max_steps=100000 use_FRM_online=False
done
wait

for seed in 0 1 2; do
python train_CoRe.py \
    env=softgym_PassWater \
    seed=$seed \
    experiment=core \
    segment=3 \
    num_train_steps=100050 num_pre_steps=9000 num_interact=5000 save_model_step=25000 \
    max_feedback=200 RRM_batch=10 RRM_update=30 resnet=1 RRM_lr=1e-4 \
    num_eval_episodes=10 save_eval_video=False \
    FRM_align_step=24975 FRM_align_max_steps=100000 use_FRM_online=False
done
wait


