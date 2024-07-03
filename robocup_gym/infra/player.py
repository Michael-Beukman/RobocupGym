import copy
import math
import re
from math import fabs
from typing import Any
import numpy as np

from robocup_gym.rl.envs.configs.data_names import ALL_JOINT_OBSERVATIONS


class Player(object):
    """
    This class was originally from [BahiaRT](https://bitbucket.org/bahiart3d/bahiart-gym/src/master/). We (WITS-FC) have since modified it extensively and changed several aspects of it; for instance, we redid the entire parsing to increase its speed. The structure of the class is largely the same.

    Perceptor Grammar:
    ```
        Grammar = (HJ (n <joint_name>) (<angle_in_degrees>))
        Example = (HJ (n raj2) (ax -15.61))
    ```
    Joint Names can be found [here](https://gitlab.com/robocup-sim/SimSpark/-/wikis/Models#physical-properties)


    This class has a method called `get_state_dict` which returns a dictionary of the current state of the player. It has the following keys:
    ```json
    {
        "HeadYaw",
        "HeadPitch",
        "LHipYawPitch",
        "RHipYawPitch",
        "LHipRoll",
        "RHipRoll",
        "LHipPitch",
        "RHipPitch",
        "LKneePitch",
        "RKneePitch",
        "LAnklePitch",
        "RAnklePitch",
        "LAnkleRoll",
        "RAnkleRoll",
        "LShoulderPitch",
        "RShoulderPitch",
        "LShoulderRoll",
        "RShoulderRoll",
        "LElbowYaw",
        "RElbowYaw",
        "LElbowRoll",
        "RElbowRoll",
        "InertialSensor/AccY",
        "InertialSensor/AccX",
        "InertialSensor/AccZ",
        "InertialSensor/GyrY",
        "InertialSensor/GyrX",
        "InertialSensor/GyrZ",
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
        "ballpos_rel_x_orientation",
        "ballpos_rel_y_orientation",
        "ballpos_rel_z_orientation",
        "orientation",
        "timesteps",
        "VEL_HeadYaw",
        "VEL_HeadPitch",
        "VEL_LHipYawPitch",
        "VEL_RHipYawPitch",
        "VEL_LHipRoll",
        "VEL_RHipRoll",
        "VEL_LHipPitch",
        "VEL_RHipPitch",
        "VEL_LKneePitch",
        "VEL_RKneePitch",
        "VEL_LAnklePitch",
        "VEL_RAnklePitch",
        "VEL_LAnkleRoll",
        "VEL_RAnkleRoll",
        "VEL_LShoulderPitch",
        "VEL_RShoulderPitch",
        "VEL_LShoulderRoll",
        "VEL_RShoulderRoll",
        "VEL_LElbowYaw",
        "VEL_RElbowYaw",
        "VEL_LElbowRoll",
        "VEL_RElbowRoll",
        "VEL_InertialSensor/AccY",
        "VEL_InertialSensor/AccX",
        "VEL_InertialSensor/AccZ",
        "VEL_InertialSensor/GyrY",
        "VEL_InertialSensor/GyrX",
        "VEL_InertialSensor/GyrZ",
        "VEL_leftFootResistance_x",
        "VEL_leftFootResistance_y",
        "VEL_leftFootResistance_z",
        "VEL_leftFootResistance_u",
        "VEL_leftFootResistance_v",
        "VEL_leftFootResistance_w",
        "VEL_rightFootResistance_x",
        "VEL_rightFootResistance_y",
        "VEL_rightFootResistance_z",
        "VEL_rightFootResistance_u",
        "VEL_rightFootResistance_v",
        "VEL_rightFootResistance_w",
        "VEL_ballpos_rel_x_orientation",
        "VEL_ballpos_rel_y_orientation",
        "VEL_ballpos_rel_z_orientation",
        "VEL_orientation",
        "VEL_timesteps"
    }
    ```

    Here is a brief description of the variables in this class:

    | Variable Name | Description |
    | --- | --- |
    | unum | The player's number |
    | joint_names | The names of all the joints (e.g raj1, ) |
    | steps | The number of timesteps since last reset|
    | is_fallen | Whether the player is fallen or not |
    | **Sensors**  |
    | acc | The acceleration of the player |
    | gyro | The gyroscope values of the player |
    | lf | The force perceptors of the left foot |
    | rf | The force perceptors of the right foot |
    | lf1 | The force perceptors of the left toe |
    | rf1 | The force perceptors of the right toe |
    | time | The current time of the simulation |
    | **Joint Angles**  |
    | neckYaw | The yaw of the neck |
    | neckPitch | The pitch of the neck |
    | leftShoulderPitch | The pitch of the left shoulder |
    | leftShoulderYaw | The yaw of the left shoulder |
    | leftArmRoll | The roll of the left arm |
    | leftArmYaw | The yaw of the left arm |
    | leftHipYawPitch | The yaw pitch of the left hip |
    | leftHipRoll | The roll of the left hip |
    | leftHipPitch | The pitch of the left hip |
    | leftKneePitch | The pitch of the left knee |
    | leftFootPitch | The pitch of the left foot |
    | leftFootRoll | The roll of the left foot |
    | rightHipYawPitch | The yaw pitch of the right hip |
    | rightHipRoll | The roll of the right hip |
    | rightHipPitch | The pitch of the right hip |
    | rightKneePitch | The pitch of the right knee |
    | rightFootPitch | The pitch of the right foot |
    | rightFootRoll | The roll of the right foot |
    | rightShoulderPitch | The pitch of the right shoulder |
    | rightShoulderYaw | The yaw of the right shoulder |
    | rightArmRoll | The roll of the right arm |
    | rightArmYaw | The yaw of the right arm |
    | leftToePitch | The pitch of the left toe |
    | rightToePitch | The pitch of the right toe |
    """

    DT = 0.02  # time delta
    INVERSE_DT = 1 / DT

    def __init__(self, unum: int, should_process_toes: bool = False):
        # should_process_toes Should only be true if the agent has toes.
        self.unum = unum
        self.should_process_toes = should_process_toes

        self.joint_names = [
            "raj1",
            "raj2",
            "raj3",
            "raj4",
            "laj1",
            "laj2",
            "laj3",
            "laj4",
            "rlj1",
            "rlj2",
            "rlj3",
            "rlj4",
            "rlj5",
            "rlj6",
            "llj1",
            "llj2",
            "llj3",
            "llj4",
            "llj5",
            "llj6",
        ]
        self.hingejoint_regex = (
            "\(HJ \(n hj1\) \(ax (.*?)\)\)\(HJ \(n hj2\) \(ax (.*?)\)\)"
            + ".*?"
            + ".*?".join([f"\(HJ \(n {myname}\) \(ax (.*?)\)\)" for myname in self.joint_names])
        )
        self.CACHE = {}

        self.reset()

    def reset(self):
        """This Resets the player to its default state."""
        self.num_times_fall = 0

        self.steps = 0
        # Standing of Fallen State
        self.is_fallen = False

        # ACC / GYR
        self.acc = [0, 0, 0]
        self.gyro = [0, 0, 0]

        # Force Perceptors of the feet
        self.lf = [[0, 0, 0], [0, 0, 0]]
        self.rf = [[0, 0, 0], [0, 0, 0]]

        # NAO TOE ONLY
        self.lf1 = []
        self.rf1 = []

        # Time
        self.time = None

        # Joints
        self.neckYaw = 0.0
        self.neckPitch = 0.0
        self.leftShoulderPitch = 0.0
        self.leftShoulderYaw = 0.0
        self.leftArmRoll = 0.0
        self.leftArmYaw = 0.0
        self.leftHipYawPitch = 0.0
        self.leftHipRoll = 0.0
        self.leftHipPitch = 0.0
        self.leftKneePitch = 0.0
        self.leftFootPitch = 0.0
        self.leftFootRoll = 0.0
        self.rightHipYawPitch = 0.0
        self.rightHipRoll = 0.0
        self.rightHipPitch = 0.0
        self.rightKneePitch = 0.0
        self.rightFootPitch = 0.0
        self.rightFootRoll = 0.0
        self.rightShoulderPitch = 0.0
        self.rightShoulderYaw = 0.0
        self.rightArmRoll = 0.0
        self.rightArmYaw = 0.0

        # NAO TOE ONLY
        self.leftToePitch = 0.0
        self.rightToePitch = 0.0

        self.max = 0

        self.prev_orientation = None
        self.my_orientation = 0
        self.orientation_velocity = 0.0

        self.player_pos = None
        self.real_ball_pos = None  # the actual ball position
        self.prev_ball_pos = None
        self.prev_velocities = [None, None, None]

        self.ball_velocity = np.array([0.0, 0.0, 0.0])
        self.ball_acceleration = np.array([0.0, 0.0, 0.0])
        self.ball_speed = 0

        self.past_dict_state = None
        self.current_dict_state = None
        self.current_velocity_dict = None

        self.joint_vector = np.zeros(22 + self.should_process_toes * 2)

    def update_player_stats(self, agent_msg: str) -> None:
        """Given a server message, all information is parsed from it and the current state of this object is updated.

        Args:
            agent_msg (str): _description_
        """
        if "mypos" in agent_msg:
            self.player_pos = self._parse_position_regex(agent_msg, "mypos")

        if "myorien" in agent_msg:
            is_bad = self.prev_orientation is None
            self.prev_orientation = copy.deepcopy(self.my_orientation)
            self.my_orientation = self._parse_orientation_regex(agent_msg, self.my_orientation) * math.pi / 180.0
            if not is_bad and self.prev_orientation is not None:
                self.orientation_velocity = (self.my_orientation - self.prev_orientation) * self.INVERSE_DT

        if "ballpos" in agent_msg:
            self.prev_ball_pos = copy.deepcopy(self.real_ball_pos)
            self.real_ball_pos = self._parse_position_regex(agent_msg, "ballpos")
            if self.prev_ball_pos is not None:
                self.ball_velocity = (self.real_ball_pos - self.prev_ball_pos) * self.INVERSE_DT
                self.prev_velocities.append(self.ball_velocity)
                self.prev_velocities = self.prev_velocities[-3:]

                # Calculate acceleration using the last three velocities and a second order approximation
                # acc = (self.prev_velocities[-1] - self.prev_velocities[-3]) / (self.DT * 2)
                vnow, vprev = self.prev_velocities[-1], self.prev_velocities[-2]
                if vprev is not None and self.steps > 4:
                    acc = (vnow - vprev) / (self.DT)
                    self.ball_acceleration = acc

                self.ball_speed = np.linalg.norm(self.ball_velocity)

        # All of the joints
        if 1:
            output = self._re(self.hingejoint_regex, agent_msg)
            assert output is not None
            g = output.groups()
            vals = [float(g[i]) for i in range(len(self.joint_names) + 2)]

            self.neckYaw = vals[0]
            self.neckPitch = vals[1]
            self.rightShoulderPitch = vals[2]
            self.rightShoulderYaw = vals[3]
            self.rightArmRoll = vals[4]
            self.rightArmYaw = vals[5]
            self.leftShoulderPitch = vals[6]
            self.leftShoulderYaw = vals[7]
            self.leftArmRoll = vals[8]
            self.leftArmYaw = vals[9]

            self.rightHipYawPitch = vals[10]
            self.rightHipRoll = vals[11]
            self.rightHipPitch = vals[12]
            self.rightKneePitch = vals[13]
            self.rightFootPitch = vals[14]
            self.rightFootRoll = vals[15]

            self.leftHipYawPitch = vals[16]
            self.leftHipRoll = vals[17]
            self.leftHipPitch = vals[18]
            self.leftKneePitch = vals[19]
            self.leftFootPitch = vals[20]
            self.leftFootRoll = vals[21]

        if self.should_process_toes:
            self.leftToePitch = self._parse_hinge_joint_regex(agent_msg, "llj7")
            self.rightToePitch = self._parse_hinge_joint_regex(agent_msg, "rlj7")
            self.lf1 = self._parse_foot_resistance_regex(agent_msg, "lf1")
            self.rf1 = self._parse_foot_resistance_regex(agent_msg, "rf1")

        # ACC/GYR
        self.acc = self._parse_acc_gyro_regex(agent_msg, "ACC", self.acc)
        self.gyro = self._parse_acc_gyro_regex(agent_msg, "GYR", self.gyro) / 180 * math.pi

        # TIME
        self.time = self._parse_time_from_regex(agent_msg, self.time)

        # FORCE RESISTANCE PERCEPTORS
        self.lf = self._parse_foot_resistance_regex(agent_msg, "lf")
        self.rf = self._parse_foot_resistance_regex(agent_msg, "rf")

        # CHECK IF PLAYER IS FALLEN
        self.is_fallen = self.check_fallen()
        self.steps += 1

        new_dicts = self._get_inner_dict_state()
        vel_dict = {}
        # compute velocities
        for k, dic in new_dicts.items():
            vel_dict[k] = {}
            for k2, v in dic.items():
                if self.current_dict_state is None:
                    vel_dict[k][k2] = 0.0
                else:
                    vel_dict[k][k2] = (v - self.current_dict_state[k][k2]) * self.INVERSE_DT

        # Override the orientation's velocity
        vel_dict["pos"]["orientation"] = self.orientation_velocity

        self.current_dict_state = new_dicts
        self.current_velocity_dict = vel_dict

        self.joint_vector = np.array([self.current_dict_state["joints"][k] for k in ALL_JOINT_OBSERVATIONS])

    def check_fallen(self) -> bool:
        """Returns true if the player has fallen.

        Returns:
            bool: Whether the player has fallen or not.
        """
        fallen = False

        X_ACEL = self.acc[0]
        Y_ACEL = self.acc[1]
        Z_ACEL = self.acc[2]

        if (fabs(X_ACEL) > Z_ACEL or fabs(Y_ACEL) > Z_ACEL) and Z_ACEL < 5:
            if (Y_ACEL < -6.5 and Z_ACEL < 3) or (Y_ACEL > 7.5 and Z_ACEL < 3) or (fabs(X_ACEL) > 6.5):
                fallen = True
                self.num_times_fall += 1
        else:
            self.num_times_fall = 0
        return fallen

    def get_state_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary of the current state of the player
        Returns:
            dict[str, Any]: The current state of the player
        """
        flat = {}
        for _, dic in self.current_dict_state.items():
            for k, v in dic.items():
                flat[k] = v
        for _, dic in self.current_velocity_dict.items():
            for k, v in dic.items():
                flat["VEL_" + k] = v
        return flat

    # ======================================= Inner State Dict ======================================= #

    def _get_inner_dict_state(self):
        if self.real_ball_pos is None or self.player_pos is None:
            ball_relative = np.array([0, 0, 0])
        else:
            ball_relative = self.real_ball_pos - self.player_pos
        orientation = self.my_orientation

        ball_x, ball_y = ball_relative[0], ball_relative[1]
        ball_x_rotated = ball_x * math.cos(orientation) - ball_y * math.sin(orientation)
        ball_y_rotated = ball_x * math.sin(orientation) + ball_y * math.cos(orientation)

        return {
            "joints": {
                "HeadYaw": self.neckYaw,
                "HeadPitch": self.neckPitch,
                "LHipYawPitch": self.leftHipYawPitch,
                "RHipYawPitch": self.rightHipYawPitch,
                "LHipRoll": self.leftHipRoll,
                "RHipRoll": self.rightHipRoll,
                "LHipPitch": self.leftHipPitch,
                "RHipPitch": self.rightHipPitch,
                "LKneePitch": self.leftKneePitch,
                "RKneePitch": self.rightKneePitch,
                "LAnklePitch": self.leftFootPitch,
                "RAnklePitch": self.rightFootPitch,
                "LAnkleRoll": self.leftFootRoll,
                "RAnkleRoll": self.rightFootRoll,
                "LShoulderPitch": self.leftShoulderPitch,
                "RShoulderPitch": self.rightShoulderPitch,
                "LShoulderRoll": self.leftShoulderYaw,
                "RShoulderRoll": self.rightShoulderYaw,
                "LElbowYaw": self.leftArmRoll,
                "RElbowYaw": self.rightArmRoll,
                "LElbowRoll": self.leftArmYaw,
                "RElbowRoll": self.rightArmYaw,
            },
            "sensors": {
                "InertialSensor/AccY": self.acc[1],
                "InertialSensor/AccX": self.acc[0],
                "InertialSensor/AccZ": self.acc[2],
                "InertialSensor/GyrY": self.gyro[1],
                "InertialSensor/GyrX": self.gyro[0],
                "InertialSensor/GyrZ": self.gyro[2],
            },
            "foot_resistance": {
                "leftFootResistance_x": self.lf[0][0],
                "leftFootResistance_y": self.lf[0][1],
                "leftFootResistance_z": self.lf[0][2],
                "leftFootResistance_u": self.lf[1][0],
                "leftFootResistance_v": self.lf[1][1],
                "leftFootResistance_w": self.lf[1][2],
                "rightFootResistance_x": self.rf[0][0],
                "rightFootResistance_y": self.rf[0][1],
                "rightFootResistance_z": self.rf[0][2],
                "rightFootResistance_u": self.rf[1][0],
                "rightFootResistance_v": self.rf[1][1],
                "rightFootResistance_w": self.rf[1][2],
            },
            "pos": {
                "ballpos_rel_x_orientation": ball_x_rotated,
                "ballpos_rel_y_orientation": ball_y_rotated,
                "ballpos_rel_z_orientation": ball_relative[2],
                "orientation": orientation,
            },
            "timesteps": {"timesteps": self.steps},
        }

    # ======================================= Parsing Code ======================================= #

    def _parse_position_regex(self, message: str, name_to_search_for: str = "ballpos") -> np.ndarray:
        """This returns an array representing the position of a particular object, with name `name_to_search_for`
            For instance, if the message is "(ballpos) (10 20 30)" and name_to_search_for="ballpos", this will return [10, 20, 30]

        Args:
            message (str): The server message
            name_to_search_for (str, optional): The name of the object to seerch for. Defaults to "ballpos".

        Returns:
            np.ndarray:
        """
        return np.array(
            # list(map(float, re.search(f"\({name_to_search_for} (.+? .+? .+?)\)", message).groups()[0].split(" ")))
            list(map(float, self._re(f"\({name_to_search_for} (.+? .+? .+?)\)", message).groups()[0].split(" ")))
        )

    def _parse_hinge_joint_regex(self, message: str, name: str, oldval: float) -> float:
        """This returns the value of a specific hingejoint value with name `name` from the message. If the message does not contain the hingejoint, it returns the old value.
        For instance, if the message is `(n hj1) (ax 10)`, then this will return 10.

        Args:
            message (str):
            name (str):
            oldval (float):

        Returns:
            float:
        """
        # ans = re.search(f"\(n {name}\) \(ax (.+?)\)", message)
        ans = self._re(f"\(n {name}\) \(ax (.+?)\)", message)
        if ans is None:
            return oldval
        return float(ans.group(1))

    def _parse_acc_gyro_regex(self, message: str, name: str, oldval: np.ndarray) -> np.ndarray:
        """This parses the acc or gyro values from the message. If the message does not contain the acc or gyro values, it returns the old value.


        Args:
            message (str):
            name (str): Either `gyr` or `acc`.
            oldval (np.ndarray):

        Returns:
            np.ndarray:
        """
        temp2 = "rt" if name.lower() == "gyr" else "a"
        # ans = re.search(f"\({name} \(n torso\) \({temp2} (.+?) (.+?) (.+?)\)", message)
        ans = self._re(f"\({name} \(n torso\) \({temp2} (.+?) (.+?) (.+?)\)", message)
        if ans is None:
            return oldval
        ans = ans.groups()
        return np.array([float(ans[0]), float(ans[1]), float(ans[2])])

    def _parse_time_from_regex(self, message: str, oldval: float) -> float:
        """Reads the time of the server from the message, and uses the old value if the message does not contain the time.

        Args:
            message (str):
            oldval (float):

        Returns:
            float:
        """
        # ans = re.search(f"\(t (.+?)\)", message)
        ans = self._re(f"\(t (.+?)\)", message)
        if ans is None:
            return oldval
        return float(ans.group(1))

    def _parse_orientation_regex(self, message: str, oldval: float) -> float:
        """Returns the orientation of the player from the message. If the message does not contain the orientation, it returns the old value.

        Args:
            message (str):
            oldval (float):

        Returns:
            float:
        """
        # ans = re.search(f"\(myorien (.+?)\)", message)
        ans = self._re(f"\(myorien (.+?)\)", message)
        if ans is None:
            return oldval
        return float(ans.group(1))

    def _parse_foot_resistance_regex(self, message: str, which_foot: str) -> np.ndarray:
        """This returns the foot resistance position and force vectors for a particular foot (`which_foot`) in a vector of size 2x3. If the message does not contain the foot resistance, it returns a vector of zeros.

        Args:
            message (str):
            which_foot (str):

        Returns:
            np.ndarray: Shape (2, 3)
        """
        default = np.zeros((2, 3), dtype=np.float32)
        ans = self._re(f"\(n {which_foot}\) \(c (.+?) (.+?) (.+?)\) \(f (.+?) (.+?) (.+?)\)", message)
        if ans is None:
            return default
        ans = ans.groups()
        if len(ans) == 0:
            return default
        return np.array(
            [float(ans[0]), float(ans[1]), float(ans[2]), float(ans[3]), float(ans[4]), float(ans[5])]
        ).reshape(2, 3)

    def _re(self, regex, str):
        if regex in self.CACHE:
            t = self.CACHE[regex]
        else:
            t = re.compile(regex)
            self.CACHE[regex] = t
        return t.search(str)
