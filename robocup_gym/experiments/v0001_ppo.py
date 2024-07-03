import fire
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from robocup_gym.rl.envs.tasks.arm_up import EnvArmUp
from robocup_gym.rl.envs.tasks.benchmark.simple_kick import EnvSimpleKick
from robocup_gym.rl.envs.tasks.benchmark.velocity_kick import KickVelocityReward
from robocup_gym.infra.utils import do_all_seeding, killall
from robocup_gym.rl.envs.configs.default import create_good_minimal_config
from robocup_gym.rl.base_rl_loop import run_experiment


def main(
    agent_type: int = 0,
    seed: int = 0,
    env_name="SimpleKick",
    timesteps: int = 40,
    # How long to train for
    n_env_steps_total: int = 5_000_000,
    # PPO Environment hyperparameters; these work reasonably well on a 128 core machine
    n_env_procs: int = 16,
    batch_size: int = 128,
    n_steps: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 10,
    ent_coef: float = 0.0,
    lr: float = 1e-4,
    net_depth: int = 2,
    net_width: int = 256,
    use_sde: bool = True,
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
    # Kill all possibly running rcssserver3d processes, and seed numpy/torch.
    killall()
    do_all_seeding(seed)

    # Create the env config
    conf = create_good_minimal_config(timesteps, clip_value, norm_mode, noise_a, noise_o)

    # Experiment name
    s = f"t{agent_type}_lr{lr}_env{env_name}_{clip_value}_{norm_mode}_ts{timesteps}_ws{wait_steps}_ec{ent_coef}_sd{use_sde}_ns{n_steps}_np{n_env_procs}"
    SSS = "v0002_ppo"
    group_name = f"{SSS}_{s}"
    name = f"{SSS}_s{seed}_{extra}_{s}"

    env_classes = {
        "SimpleKick": EnvSimpleKick,
        "ArmUp": EnvArmUp,
        "VelocityKick": KickVelocityReward,
    }
    assert env_name in env_classes, f"Unknown env_name: {env_name}, expected one of {env_classes.keys()}"
    env_cls = env_classes[env_name]

    env_kwargs = {
        "env_config": conf,
        "sleep_time_after_proc_starts": sleep_time_after_starts,
        "wait_steps": wait_steps,
        "agent_type": agent_type,
    }
    run_experiment(
        name,
        env_cls,
        env_kwargs=env_kwargs,
        n_env_procs=n_env_procs,
        agent_kwargs=dict(
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            learning_rate=lr,
            policy_kwargs=dict(net_arch=[net_width] * net_depth),
            use_sde=use_sde,
            clip_range_vf=1.0,
        ),
        n_steps=n_env_steps_total,
        vec_env_class=SubprocVecEnv,
        AGENT_CLASS=PPO,
        make_vec_env_kwargs=dict(wrap_in_vecnormalise=False),
        wandb_kwargs=dict(group_name=group_name),
        eval_callback_env=lambda: EnvSimpleKick(
            vectorise_index=max(n_env_procs + 1, 500), **(env_kwargs | {"wait_steps": 200})
        ),  # longer wait time.
    )

    killall()


if __name__ == "__main__":
    fire.Fire(main)
