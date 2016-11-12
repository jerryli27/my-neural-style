def get_global_step_from_save_dir(save_dir):
    return int(save_dir[save_dir.rfind("-")+1:])