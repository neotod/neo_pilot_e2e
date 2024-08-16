import os
from dotenv import load_dotenv

load_dotenv()


IMAGE_WIDTH = 256  # input image (from the camera sensor) width
CAR_MIDDLE_X = IMAGE_WIDTH // 2

# algorithms constants
VERTICAL_LINE_MIN_SLOPE = 0.4
HORIZ_LINE_MAX_SLOPE = 0.2
OBSTABLE_DETECTION_THRESHOLD = 20
HORIZ_LINE_DETECTION_THRESHOLD = 5

# speeds
NORMAL_STATE_SPEED = int(os.getenv('NORMAL_STATE_SPEED')) # 30 in simulation

print('norm speed')
print(NORMAL_STATE_SPEED)

IGNITE_SPEED = NORMAL_STATE_SPEED + 30 # 5 in simulation
TURN_STATE_SPEED = 30 # 15 in simulation
GO_BACK_STATE_SPEED = -NORMAL_STATE_SPEED # 15 in simulation
CAR_STOP_REVERSE_SPEED = -60

# steering
MAX_ANGLE = 15
MIN_ANGLE = -18 

# digital PID coeffs
KP = 1
KI = 0.5
KD = 0.1

WINDUP_MAX = 7


# time constants
INTERSECTION_WAIT_TIME = 0.5
TURN_WAIT_TIME = 0.5
OBSTACLE_WAIT_TIME = 3


# logging
LOG_SERVER_EVERY = 30


