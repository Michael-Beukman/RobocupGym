D=`dirname "$0"`
source $D/env.sh
singularity exec  $SING_ARGS $CONTAINER_PATH rcssserver3d $@
