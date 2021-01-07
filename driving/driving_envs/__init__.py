from gym.envs.registration import register


register(id="ContinuousSparse-v0", entry_point="driving_envs.envs:GridworldSparseEnv")
register(id="Continuous-v0", entry_point="driving_envs.envs:GridworldContinuousEnv")
register(id="ContinuousMultiObjLL-v0", entry_point="driving_envs.envs:GridworldContinuousMultiObjLLEnv")
register(id="ContinuousMultiObjRR-v0", entry_point="driving_envs.envs:GridworldContinuousMultiObjRREnv")
register(id="ContinuousMultiObjLR-v0", entry_point="driving_envs.envs:GridworldContinuousMultiObjLREnv")
register(id="ContinuousMultiObjRL-v0", entry_point="driving_envs.envs:GridworldContinuousMultiObjRLEnv")
register(id="ContinuousNoneLL-v0", entry_point="driving_envs.envs:GridworldContinuousNoneLLEnv")
register(id="ContinuousNoneRR-v0", entry_point="driving_envs.envs:GridworldContinuousNoneRREnv")
register(id="ContinuousNoneLR-v0", entry_point="driving_envs.envs:GridworldContinuousNoneLREnv")
register(id="ContinuousNoneRL-v0", entry_point="driving_envs.envs:GridworldContinuousNoneRLEnv")
register(id="ContinuousAdjustLR-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustLREnv")
register(id="ContinuousAdjustRL-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustRLEnv")
register(id="ContinuousAdjustRLR-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustRLREnv")
register(id="ContinuousAdjustRRL-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustRRLEnv")
register(id="ContinuousAdjustLL-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustLLEnv")
register(id="ContinuousAdjustRR-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustRREnv")
register(id="ContinuousAdjustRLL-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustRLLEnv")
register(id="ContinuousAdjustRRR-v0", entry_point="driving_envs.envs:GridworldContinuousAdjustRRREnv")

