# Relaxing Stage (take the snapshot with 100 updates)
python main.py --env-name HomotopyRelaxUpAnt-v0 --batch-size 50000 --output_dir snapshot/HomotopyTransferRelaxUp --save-interval 5 --snapshot_path snapshot/down.tar

# Barrier Set Size method
python main_homotopy.py --env-name HomotopyObstacleUpAnt-v0 --batch-size 50000 --output_dir snapshot/HomotopyTransferObstacleUpAnt_from100 --save-interval 5 --snapshot_path snapshot/up_relax_100.tar

# Reward Weight method
python main_homotopy1.py --env-name HomotopyRewardUpAnt-v0 --batch-size 50000 --output_dir snapshot/HomotopyTransferRewardUpAnt_from100 --save-interval 5 --snapshot_path snapshot/up_relax_100.tar

# Compute Iteration num
python compute_iter_num.py snapshot/HomotopyTransferObstacleUpAnt_from100
python compute_iter_num.py snapshot/HomotopyTransferRewardUpAnt_from100
