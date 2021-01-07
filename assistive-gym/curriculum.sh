## Learn the source policy
python train.py --env-name "FeedingPR2HomotopyDown-v0" --num-env-steps 20000000 --save-dir ./up

## Relaxing Stage
python train.py --env-name "FeedingPR2HomotopyDownNo-v0" --num-env-steps 2000000 --save-dir ./up_no --load-policy pretrained/up.pt

## Curriculum Learning Stage
python train_curriculum_obs.py --env-name "FeedingPR2HomotopyDownAdjust-v0" --num-env-steps 2000000 --save-dir ./up2down --load-policy pretrained/up_no.pt --obs_size 1.0 --rew_factor 1.0 --eval-interval 10

## Running direct fine-tuning baseline
python train.py --env-name "FeedingPR2HomotopyDown-v0" --num-env-steps 2000000 --save-dir ./up2down_direct --eval-interval 10 --load-policy pretrained/up.pt

## Running pnn baseline
python train_pnn.py --env-name "FeedingPR2HomotopyDown-v0" --num-env-steps 2000000 --save-dir ./up2down_pnn --eval-interval 10 --load-policy pretrained/up.pt

## Running l2sp baseline
python train_l2sp.py --env-name "FeedingPR2HomotopyDown-v0" --num-env-steps 2000000 --save-dir ./up2down_l2sp --eval-interval 10 --load-policy pretrained/up.pt

## Runing bss baseline
python train_bss.py --env-name "FeedingPR2HomotopyDown-v0" --num-env-steps 2000000 --save-dir ./up2down_bss --eval-interval 10 --load-policy pretrained/up.pt
