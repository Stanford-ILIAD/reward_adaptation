import os

if __name__ == '__main__':
    model_paths=[ ("weight_-1", "best_model_151040_[710.741].pkl"),
    ("weight_-0.5", "best_model_1428480_[309.0573].pkl"),
    ("weight_0", "best_model_87040_[-37.172234].pkl"),
    ("weight_0.5", "best_model_16640_[-33.3783].pkl"),
    ("weight_1", "best_model_8960_[-29.448769].pkl"),
    ("weight_2", "best_model_1980160_[1087.1274].pkl"),
    ("weight_4", "best_model_1827840_[2262.8826].pkl"),
    ("weight_6", "best_model_2670080_[3432.1975].pkl"),
    ("weight_8", "best_model_3175680_[4600.899].pkl"),
    ("weight_10", "best_model_3527680_[5769.4253].pkl"),
    ("weight_100", "best_model_14080_[58647.805].pkl"),]
    model_paths = [os.path.join('reward_curriculum_expts', item[0], item[1]) for item in model_paths]
    os.system('tar -zcf trained_models.tar.gz ' + " ".join(model_paths))

