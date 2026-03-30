def train_model(*args, **kwargs):
    from retrainer_runtime_patch import _resume_ppo_train_model
    return _resume_ppo_train_model(*args, **kwargs)
