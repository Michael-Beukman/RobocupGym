import glob
import os
import subprocess

import fire
from robocup_gym.config.vars import CONDA_ENV_NAME, ROBOCUP_GYM_ROOT, LOG_DIR, SLURM_DIR
from robocup_gym.infra.utils import get_date, path


def main(python_filename: str, partition_name: str = "stampede", use_slurm: bool = True, args=""):
    """This creates a slurm file and runs it

    Args:
        partition_name (str): Partition to run the code on
        yaml_config_file (str): The config file to use for everything
        use_slurm (bool): If true, uses slurm, otherwise executes the script with bash
    """
    date = get_date()
    local = not use_slurm
    # hashes = conf.hash(False, False)

    python_name = python_filename
    # clean_name = f"{hashes}-{}-{date}"
    # Create Slurm File
    pname = python_name.replace("/", "_")  # + "__" + args.replace(" ", '_')
    my_dir = os.path.join(LOG_DIR, "slurms")
    LOG_FILE = os.path.join(my_dir, f"slurm_log_{pname}")

    os.makedirs(my_dir, exist_ok=True)

    OTHER_EXPORTS = ""
    ALL_RUN_VALUES = f"time ./run.sh {python_name} {args}"
    print(f"Want to run {ALL_RUN_VALUES}")
    pname_jobname = pname
    pname_jobname = pname_jobname.replace("kudu_gym_", "")
    pname_jobname = pname_jobname.replace("experiments_", "")
    pname_jobname = pname_jobname.replace("mike_", "")
    pname_jobname = pname_jobname.replace("kicks_", "")
    s = f"""#!/bin/bash
#SBATCH -p {partition_name}
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J {pname_jobname}
#SBATCH -o {LOG_FILE}.%N.%j.out

source ~/.bashrc
trap "echo Signal USR1 received!; killall -9 python; sleep 1; killall -9 rcssserver3d agentspark;" USR1
cd {ROBOCUP_GYM_ROOT}
conda activate {CONDA_ENV_NAME}
{OTHER_EXPORTS}
{ALL_RUN_VALUES} &
wait;
sleep 5
killall -9 rcssserver3d agentspark
"""
    dir = path(SLURM_DIR)
    fpath = os.path.join(dir, f"{date}_{pname}.slurm")
    with open(fpath, "w+") as f:
        f.write(s)

    # Run it
    if use_slurm:
        # ,mscluster47,mscluster48,mscluster49,mscluster50,mscluster51,mscluster52,
        ans = subprocess.call(
            f"sbatch --exclude mscluster42,mscluster75,mscluster53,mscluster54,mscluster55,mscluster56,mscluster57,mscluster58,mscluster59,mscluster60,mscluster61,mscluster62,mscluster63 --signal=B:USR1@300 {fpath}".split(
                " "
            )
        )
    else:
        print(f"Logging to {LOG_FILE}, running {fpath}")
        ans = subprocess.Popen(f"bash {fpath} 2>&1".split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ans2 = subprocess.call(f"tee {LOG_FILE}.local.out".split(" "), stdin=ans.stdout)
        print("ANS2", ans2)
        ans.wait()
        ans = ans.returncode
    assert ans == 0
    print("Successfully Ran")


if __name__ == "__main__":
    fire.Fire(main)
