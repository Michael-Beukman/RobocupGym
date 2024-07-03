from typing import Any, Dict
import wandb
import glob


def init_wandb(exp_name: str, config: Dict[str, Any], project_name: str, group_name: str = None) -> wandb.run:
    if group_name is None:
        group_name = exp_name
    dic = dict(
        project=project_name,
        name=exp_name,
        config=config,
        job_type="train",  # exp_name,
        tags=["robocup", "kick"],
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
    )
    # Properly sets up wandb
    run_id = f"{exp_name}_1"
    resume = "allow"
    return wandb.init(id=run_id, resume=resume, **dic)
