import time

import fire
import numpy as np
from robocup_gym.rl.envs.configs.default import (
    create_good_minimal_config,
)
from robocup_gym.rl.envs.tasks.evaluation_env import EnvKickEvaluation
from robocup_gym.infra.utils import killall
from robocup_gym.rl.utils import evaluate_agent, get_latest_checkpoint


def analyse_good_eval_env(
    checkpoint_to_eval: str, timesteps: int = 40, clip_value=1.0, norm_mode="min_max_analytic", agent_type: int = 0
):
    killall()
    time.sleep(1)
    args = {}
    kwargs = {}
    conf = create_good_minimal_config(timesteps, clip_value, norm_mode, 0.0, 0.0)
    conf.player_start_pos = (-13.2, 0.03 + 0.0258778, 0.375)
    conf.ball_start_pos = (-13.0, 0.0, 0.044)
    ## End
    ENV_CLASS = EnvKickEvaluation
    time.sleep(1)
    check = get_latest_checkpoint(checkpoint_to_eval) if "save_models" not in checkpoint_to_eval else checkpoint_to_eval
    mean, std, env = evaluate_agent(
        ENV_CLASS,
        model_path=check,
        env_kwargs={"env_config": conf, "wait_steps": 400, "agent_type": agent_type, "do_only_x": True} | args,
        total_eps=50,
        verbose=False,
        return_env=True,
        **kwargs,
    )
    TT = env.get_x_y_z_pos()
    xx, yy = (
        f"{np.round(TT[0][0], 2):<7} ({np.round(TT[0][1], 2)})",
        f"{np.round(TT[1][0], 2):<7} ({np.round(TT[1][1], 2)})",
    )
    print(f"Model ({checkpoint_to_eval}; {check}): {mean} +- {std} || {xx} {yy}")
    killall()


if __name__ == "__main__":
    fire.Fire(analyse_good_eval_env)
