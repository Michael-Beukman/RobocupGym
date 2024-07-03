import fire
from robocup_gym.rl.envs.configs.default import GoodConfigVelocitiesMinimal
from robocup_gym.infra.utils import killall
from robocup_gym.rl.base_rl_loop import run_experiment
from robocup_gym.rl.envs.tasks.env_simple_kick import EnvSimpleKick


def main(n_env_procs: int = 1):
    killall()
    conf = GoodConfigVelocitiesMinimal
    conf.options.max_number_of_timesteps = 20
    run_experiment(
        f"v0001_minimal_{n_env_procs}",
        EnvSimpleKick,
        env_kwargs={"env_config": conf, "sleep_time_after_proc_starts": 20},
        n_env_procs=n_env_procs,
    )


if __name__ == "__main__":
    fire.Fire(main)
