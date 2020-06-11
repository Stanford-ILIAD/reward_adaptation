# PRE-TRAINED POLICIES


## L->R ##
# Source L policies (in relaxed setting)
BL_s = ["output/fetch2", "BL_v02", "best_model_10000_10.418350900489902.pkl"]
BL_s1 = ["output/fetch3", "BL", "best_model_40000_12.044709828579684.pkl"]
BL_s2 = ["output/fetch3", "BL", "best_model_10000_10.731997219838156.pkl"]

# Fine tune
BL_BR = ["output/fetch2", "BL_v02_BR", "best_model_170000_-2.1941347378969267.pkl"]
BL_BR1 = ["output/fetch3", "BL_BR1", "best_model_220000_-3.328827578282447.pkl"]
BL_BR2 = ["output/fetch3", "BL_BR2", "best_model_10000_-3.1450411205427806.pkl"]

# Ours:
# L -> 0 (relaxing stage)
BL_BR0 = ["output/fetch2", "BL_v02_BR0", "best_model_300000_10.615581360706914.pkl"]
BL_BR01 = ["output/fetch3", "BL_BR01", "best_model_10_10.615581360706907.pkl"]
BL_BR02 = ["output/fetch2", "BL_v02_BR0", "best_model_320000_10.859693369490124.pkl"]
# L -> 0 -> R (use these ones)
BL_BR0_BR = ["output/fetch2", "BL_v02_BR0_BR", "best_model_40000_10.286005362445314.pkl"]
BL_BR0_BR1 = ["output/fetch3", "BL_B0R_BR1", "best_model_40000_10.286005362445314.pkl"]
BL_BR0_BR2 = ["output/fetch3", "B1L_B0R_BR2", "best_model_40000_10.28486653681686.pkl"]

# L2SP
BL_BR_L2SP = ["output/fetch_L2SP", "BL_BR_L2SP", "best_model_70000_-3.1596475571534803.pkl"]
BL_BR_L2SP1 = ["output/fetch_L2SP", "BL_BR_L2SP1", "best_model_430000_-3.5892668665072325.pkl"]
BL_BR_L2SP2 = ["output/fetch_L2SP", "BL_BR_L2SP2", "best_model_420000_1.8677240270005304.pkl"]

# PNN
BL_BR_PNN = ["output/fetch_PNN", "BL_BR_PNN", "best_model_10000_-9.703680517774709.pkl"]
BL_BR_PNN1 = ["output/fetch_PNN", "BL_BR_PNN1", "best_model_10000_-9.983871471753242.pkl"]
BL_BR_PNN2 = ["output/fetch_PNN", "BL_BR_PNN2", "best_model_20000_-8.215743327777158.pkl"]

# Random init on R
BR = ["output/fetch3", "BR_final", "best_model_40000_1.0243441636602049.pkl"]
BR1 = ["output/fetch3", "BR_final1", "best_model_480000_1.9660875790152643.pkl"]
BR2 = ["output/fetch3", "BR_final2", "best_model_70000_10.568043340719738.pkl"]



## R->L ##
# Source R policies (in relaxed setting)
BR_s = ["output/fetch", "BR_v3", "best_model_100000_10.64116769685721.pkl"]
BR_s1 = ["output/fetch", "BR_v3", "best_model_90000_10.50114095298899.pkl"]
BR_s2 =["output/fetch3", "Bn1R_seed1", "best_model_160000_10.523983048169214.pkl"]

# Fine tune
BR_BL = ["output/fetch2", "BR_BL", "best_model_420000_-3.473249559282106.pkl"]
BR_BL1 = ["output/fetch3", "BR_BL1", "best_model_460000_-3.466495971067752.pkl"]
BR_BL2 = ["output/fetch3", "BR_BL2", "best_model_440000_-0.9422334859086501.pkl"]

# Ours:
# R -> 0 (relaxing stage)
BR_BL0 = ["output/fetch2", "BR_BL0", "best_model_150000_11.456230865755833.pkl"]
# R -> 0 -> 1 (curriculum stage 1)
BR_BL0_BL1 = ["output/fetch2", "BR_BL0_BL1", "best_model_10000_8.68237885196727.pkl"]
# R -> 0 -> 1 -> 5 (curriculum stage 2)
BR_BL0_BL1_BL5 = ["output/fetch2", "BR_BL0_BL1_BL5", "best_model_90000_10.866158360182357.pkl"]
BR_BL0_BL1_BL51 = ["output/fetch2", "BR_BL0_BL1_BL5", "best_model_30000_10.789408159935599.pkl"]
BR_BL0_BL1_BL52 = ["output/fetch2", "BR_BL0_BL1_BL5", "best_model_20000_10.50922430887016.pkl"]
# R -> 0 -> 1 -> 5 -> L (use these ones)
BR_BL0_BL1_BL5_BL = ["output/fetch3", "BR_BL0_BL1_BL5_BL", "best_model_10_10.866158360182357.pkl"]
BR_BL0_BL1_BL5_BL1 = ["output/fetch3", "BR_BL0_BL1_BL5_BL1", "best_model_10_10.789408159935599.pkl"]
BR_BL0_BL1_BL5_BL2 = ["output/fetch3", "BR_BL0_BL1_BL5_BL2", "best_model_10_10.509224308870168.pkl"]

# L2SP
BR_BL_L2SP = ["output/fetch_L2SP", "BR_BL_L2SP", "best_model_370000_-3.100138831477136.pkl"]
BR_BL_L2SP1 = ["output/fetch_L2SP", "BR_BL_L2SP1", "best_model_250000_-3.0800435426586557.pkl"]
BR_BL_L2SP2 = ["output/fetch_L2SP", "BR_BL_L2SP2", "best_model_510000_-1.3970538542317223.pkl"]

# PNN
BR_BL_PNN = ["output/fetch_PNN", "BR_BL_PNN", "best_model_90000_-9.655959716539263.pkl"]
BR_BL_PNN1 = ["output/fetch_PNN", "BR_BL_PNN1", "best_model_10000_-10.497507002618553.pkl"]
BR_BL_PNN2 = ["output/fetch_PNN", "BR_BL_PNN2", "best_model_20000_-0.49018776537117903.pkl"]

# Random init on L
BL =  ["output/fetch2", "BL_final", "best_model_500000_-2.2114741296401537.pkl"]
BL1 = ["output/fetch3", "BL_final1", "best_model_30000_-3.4597753562296294.pkl"]
BL2 = ["output/fetch3", "BL_final2", "best_model_160000_-3.365003964097909.pkl"]
