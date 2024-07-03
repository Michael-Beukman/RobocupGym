D=`dirname "$0"`
source $D/.env
REPO_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"/..
CONTAINER_PATH=$REPO_PATH/apptainer/robocup_training_michael_good.sif
ENV_ARGS="--env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/.singularity.d/libs"
SING_ARGS=""

