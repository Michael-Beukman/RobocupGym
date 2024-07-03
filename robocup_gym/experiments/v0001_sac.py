import fire
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from robocup_gym.rl.envs.tasks.arm_up import EnvArmUp
from robocup_gym.rl.envs.tasks.benchmark.simple_kick import EnvSimpleKick
from robocup_gym.rl.envs.tasks.benchmark.velocity_kick import KickVelocityReward
from robocup_gym.infra.utils import do_all_seeding, get_normalisation, killall
from robocup_gym.rl.envs.configs.default import GoodConfigVelocitiesMinimal, create_good_minimal_config
from robocup_gym.rl.base_rl_loop import run_experiment


def main(
    agent_type: int = 0,
    seed: int = 0,
    env_name="SimpleKick",
    timesteps: int = 40,
    # How long to train for
    n_env_steps_total: int = 5_000_000,
    # SAC hyperparameters
    n_env_procs=128,
    lr: float = 1e-4,
    net_depth: int = 2,
    net_width: int = 256,
    clip_value: float = 1,
    # How to normalise the observations; this one works decently well.
    norm_mode="min_max_analytic",
    # How long to wait after termination but before computing the reward (e.g., for the ball to stop moving)
    wait_steps: int = 20,
    # How long to wait before starting the environment; using 128 cores, 20 or so seconds is good.
    sleep_time_after_starts: int = 20,
    # An extra string to add to the experiment name
    extra: str = "A",
    # Action and observation noise
    noise_a: float = 0.0,
    noise_o: float = 0.0,
):
    killall()
    do_all_seeding(seed)

    conf = create_good_minimal_config(timesteps, clip_value, norm_mode, noise_a, noise_o)

    s = f"t{agent_type}_env{env_name}_lr{lr}_{clip_value}_{norm_mode}_ts{timesteps}_ws{wait_steps}_n{n_env_procs}"
    env_kwargs = {
        "env_config": conf,
        "sleep_time_after_proc_starts": sleep_time_after_starts,
        "wait_steps": wait_steps,
        "agent_type": agent_type,
    }
    SSS = "v0002_sac"
    group_name = f"{SSS}_{s}"
    name = f"{SSS}_s{seed}_{extra}_{s}"
    env_classes = {
        "SimpleKick": EnvSimpleKick,
        "ArmUp": EnvArmUp,
        "VelocityKick": KickVelocityReward,
    }
    assert env_name in env_classes, f"Unknown env_name: {env_name}, expected one of {env_classes.keys()}"
    env_cls = env_classes[env_name]

    run_experiment(
        name,
        env_cls,
        env_kwargs=env_kwargs,
        n_env_procs=n_env_procs,
        n_steps=n_env_steps_total,
        vec_env_class=SubprocVecEnv,
        AGENT_CLASS=SAC,
        agent_kwargs=dict(
            learning_rate=lr,
            policy_kwargs=dict(net_arch=[net_width] * net_depth),
        ),
        make_vec_env_kwargs=dict(wrap_in_vecnormalise=False),
        wandb_kwargs=dict(group_name=group_name),
        eval_callback_env=lambda: EnvSimpleKick(
            vectorise_index=max(n_env_procs + 1, 500), **(env_kwargs | {"wait_steps": 200})
        ),  # longer wait time.
    )

    killall()


if __name__ == "__main__":
    fire.Fire(main)
