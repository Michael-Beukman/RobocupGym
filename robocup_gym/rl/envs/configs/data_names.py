ACTION_LEFT_ARM = [
    "lae1",
    "lae2",
    "lae3",
    "lae4",
]
ACTION_RIGHT_ARM = [
    "rae1",
    "rae2",
    "rae3",
    "rae4",
]

ACTION_LEFT_LEG = [
    "lle1",
    "lle2",
    "lle3",
    "lle4",
    "lle5",
    "lle6",
]

ACTION_RIGHT_LEG = ["rle1", "rle2", "rle3", "rle4", "rle5", "rle6"]

ACTION_HEAD = [
    "he1",
    "he2",
]


ACTION_ARMS_LEGS = ACTION_LEFT_ARM + ACTION_RIGHT_ARM + ACTION_LEFT_LEG + ACTION_RIGHT_LEG


OBSERVATION_JOINT_LEFT_ARM = [
    "LElbowYaw",
    "LElbowRoll",
    "LShoulderPitch",
    "LShoulderRoll",
]

OBSERVATION_JOINT_RIGHT_ARM = [
    "RElbowYaw",
    "RElbowRoll",
    "RShoulderPitch",
    "RShoulderRoll",
]
OBSERVATION_JOINT_LEFT_LEG = [
    "LAnklePitch",
    "LAnkleRoll",
    "LHipPitch",
    "LHipRoll",
    "LHipYawPitch",
    "LKneePitch",
]
OBSERVATION_JOINT_RIGHT_LEG = [
    "RAnklePitch",
    "RAnkleRoll",
    "RHipPitch",
    "RHipRoll",
    "RHipYawPitch",
    "RKneePitch",
]
OBSERVATION_JOINT_HEAD = [
    "HeadPitch",
    "HeadYaw",
]
OBSERVATION_JOINTS_ARMS_LEGS = (
    OBSERVATION_JOINT_LEFT_ARM + OBSERVATION_JOINT_RIGHT_ARM + OBSERVATION_JOINT_LEFT_LEG + OBSERVATION_JOINT_RIGHT_LEG
)

OBSERVATION_JOINTS_ARMS_LEGS_VELOCITIES = ["VEL_" + j for j in OBSERVATION_JOINTS_ARMS_LEGS]


OBSERVATION_SENSORS = [
    "InertialSensor/AccY",
    "InertialSensor/AccX",
    "InertialSensor/AccZ",
    "InertialSensor/GyrY",
    "InertialSensor/GyrX",
    "InertialSensor/GyrZ",
]

OBSERVATION_SENSORS_VEL = ["VEL_" + j for j in OBSERVATION_SENSORS]

OBSERVATION_FOOT_RES = [
    "leftFootResistance_x",
    "leftFootResistance_y",
    "leftFootResistance_z",
    "leftFootResistance_u",
    "leftFootResistance_v",
    "leftFootResistance_w",
    "rightFootResistance_x",
    "rightFootResistance_y",
    "rightFootResistance_z",
    "rightFootResistance_u",
    "rightFootResistance_v",
    "rightFootResistance_w",
]

OBSERVATION_FOOT_RES_MINIMAL = [
    "leftFootResistance_u",
    "leftFootResistance_v",
    "leftFootResistance_w",
    "rightFootResistance_u",
    "rightFootResistance_v",
    "rightFootResistance_w",
]

OBSERVATION_FOOT_RES_VEL = ["VEL_" + j for j in OBSERVATION_FOOT_RES]
OBSERVATION_FOOT_RES_MINIMAL_VEL = ["VEL_" + j for j in OBSERVATION_FOOT_RES_MINIMAL]


OBSERVATION_ORIENTATION_BALL = [
    "ballpos_rel_x_orientation",
    "ballpos_rel_y_orientation",
    "ballpos_rel_z_orientation",
]

OBSERVATION_ORIENTATION_BALL_VEL = [
    "VEL_ballpos_rel_x_orientation",
    "VEL_ballpos_rel_y_orientation",
    "VEL_ballpos_rel_z_orientation",
]

OBSERVATION_ORIENTATION = ["orientation", "VEL_orientation"]

OBSERVATION_BALL = [
    "ballpos_rel_x_orientation",
    "ballpos_rel_y_orientation",
    "ballpos_rel_z_orientation",
]

OBSERVATION_BALL_VEL = [
    # "VEL_ballpos_rel_x",
    # "VEL_ballpos_rel_y",
    # "VEL_ballpos_rel_z",
    "VEL_ballpos_rel_x_orientation",
    "VEL_ballpos_rel_y_orientation",
    "VEL_ballpos_rel_z_orientation",
]

OBSERVATION_TIMESTEPS = ["timesteps"]

OBSERVATION_BALL_DIST_ANGLE = [
    "desired_rl_kick_distance",
    "desired_rl_kick_angle_degrees",
]


OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS = (
    OBSERVATION_JOINTS_ARMS_LEGS
    + OBSERVATION_JOINTS_ARMS_LEGS_VELOCITIES
    + OBSERVATION_SENSORS
    + OBSERVATION_BALL
    + OBSERVATION_BALL_VEL
    + OBSERVATION_FOOT_RES
    + OBSERVATION_TIMESTEPS
)


OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES = (
    OBSERVATION_JOINTS_ARMS_LEGS
    + OBSERVATION_JOINTS_ARMS_LEGS_VELOCITIES
    + OBSERVATION_SENSORS
    + OBSERVATION_SENSORS_VEL
    + OBSERVATION_BALL
    + OBSERVATION_BALL_VEL
    + OBSERVATION_FOOT_RES
    + OBSERVATION_FOOT_RES_VEL
    + OBSERVATION_TIMESTEPS
)

OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_NO_BALL = (
    OBSERVATION_JOINTS_ARMS_LEGS
    + OBSERVATION_JOINTS_ARMS_LEGS_VELOCITIES
    + OBSERVATION_SENSORS
    + OBSERVATION_SENSORS_VEL
    + OBSERVATION_FOOT_RES
    + OBSERVATION_FOOT_RES_VEL
    + OBSERVATION_TIMESTEPS
)


OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION = (
    OBSERVATION_JOINTS_ARMS_LEGS
    + OBSERVATION_JOINTS_ARMS_LEGS_VELOCITIES
    + OBSERVATION_SENSORS
    + OBSERVATION_SENSORS_VEL
    + OBSERVATION_ORIENTATION_BALL
    + OBSERVATION_ORIENTATION_BALL_VEL
    + OBSERVATION_ORIENTATION
    + OBSERVATION_FOOT_RES
    + OBSERVATION_FOOT_RES_VEL
    + OBSERVATION_TIMESTEPS
)

OBSERVATION_JOINTS_SENSORS_FOOT_MINIMAL_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION = (
    OBSERVATION_JOINTS_ARMS_LEGS
    + OBSERVATION_JOINTS_ARMS_LEGS_VELOCITIES
    + OBSERVATION_SENSORS
    + OBSERVATION_SENSORS_VEL
    + OBSERVATION_ORIENTATION_BALL
    + OBSERVATION_ORIENTATION_BALL_VEL
    + OBSERVATION_ORIENTATION
    + OBSERVATION_FOOT_RES_MINIMAL
    + OBSERVATION_FOOT_RES_MINIMAL_VEL
    + OBSERVATION_TIMESTEPS
)


OBSERVATION_ALL_BALL_DIST_ANGLE = (
    OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION + OBSERVATION_BALL_DIST_ANGLE
)

OBSERVATION_ALL_BALL_DIST_ANGLE_NO_TARGET = (
    OBSERVATION_JOINTS_SENSORS_FOOT_TIMESTEPS_VELOCITIES_CORRECT_BALL_ORIENTATION
)


ALL_ACTIONS = ACTION_HEAD + ACTION_ARMS_LEGS
ALL_JOINT_OBSERVATIONS = OBSERVATION_JOINT_HEAD + OBSERVATION_JOINTS_ARMS_LEGS
