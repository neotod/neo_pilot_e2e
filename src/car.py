import numpy as np
from src import const, ConvNet, ConvNet2, ConvNet54
import time
import os
import json
import cv2
import sqlite3
import os
import datetime as dt
import Jetson.GPIO as GPIO
from PIL import Image
from dotenv import load_dotenv
import serial
import glob
import torch
from skimage import exposure

import torchvision.transforms as transforms
import torch.nn as nn


GPIO.setmode(GPIO.BOARD)

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device -> {device}')


class Car:
    def __init__(self, imgs_saving_path, steer=0, speed=0) -> None:
        self.cap = cv2.VideoCapture(
            self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER
        )
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

        self.latest_img = None
        self.imgs_saving_path = imgs_saving_path
        self.settings = {
            "logging": {"image": True, "state": True, "datetime": False},
            "movement": {"stop": False},
        }
        # logging
        self.log_server_counter = 0

        # horz line detection
        self.horiz_line_detection_counter = 0

        self.image = None
        self.sensors = None

        self.steer = steer
        self.speed = speed
        self.sensors = [0]

        self.current_side = None

        # digital PID stuff
        self.prev_error = 0
        self.prev_t = 0
        self.integral_sum = 0

        self.db_conn = sqlite3.connect(os.getenv("DB_PATH"))

        # pins
        self.PINS = {
            "ultrasonic": {"trig": 15, "echo": 16},
        }

        # ultrasonic
        self.MAX_RESPONSE_WAIT = 1  # 1ms
        GPIO.setup(self.PINS["ultrasonic"]["trig"], GPIO.OUT)
        GPIO.setup(self.PINS["ultrasonic"]["echo"], GPIO.IN)

        # communicating to arduino stuff
        arduino_path = glob.glob('/dev/ttyUSB*')[0]
        print(f'arduino port path -> {arduino_path}')
        self.arduino = serial.Serial(arduino_path, 9600, timeout=1)

        # e2e model
        print('loading the e2e model')
        # self.model = ConvNet54(input_size=(64, 128), in_channels=1, n_conv_layers=5).to(device)
        self.model = ConvNet2(input_size=(64, 128), in_channels=1, n_conv_layers=5).to(device)
        self.model.load_state_dict(torch.load(os.getenv('MODEL_SAVE_PATH')))
        self.model.eval()

        self.img_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.clahe = cv2.createCLAHE(clipLimit=2.0)



    def gstreamer_pipeline(
        self,
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=340,
        display_height=340,
        framerate=60,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink sync=false drop=true"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    def ignite(self, iters=2):
        print('igniting the car. are you ready?')
        print('igniting the e2e model...')
        for i in range(10):
            x = torch.randn(1, 1, 64, 128).to(device)

            t1 = time.time()
            y = self.model(x)
            t2 = time.time()

            print(f'{i}th inference took: {t2-t1:5f}s')

        print("LET'S GOOO!")

        for _ in range(iters):
            self.set_speed(const.IGNITE_SPEED)
            self.set_steering(0)

            self.send_arduino_data()


    def get_image(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read frame.")
        
        frame = frame[42:298, 42:298]
        self.latest_img = frame

        return frame

    def get_ultrasonic_distance(self):
        try:
            # Send a trigger pulse
            GPIO.output(self.PINS["ultrasonic"]["trig"], GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(self.PINS["ultrasonic"]["trig"], GPIO.LOW)

            # Wait for the echo to start
            start = time.time()
            pulse_start = time.time()
            while GPIO.input(self.PINS["ultrasonic"]["echo"]) == 0:
                pulse_start = time.time()

                if (time.time() - start) > self.MAX_RESPONSE_WAIT:
                    raise Exception("waiting time for ultrasonic exceeded.")

            # Wait for the echo to end
            start = time.time()
            pulse_end = time.time()
            while GPIO.input(self.PINS["ultrasonic"]["echo"]) == 1:
                pulse_end = time.time()

                if (time.time() - start) > self.MAX_RESPONSE_WAIT:
                    raise Exception("waiting time for ultrasonic exceeded.")

            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (
                pulse_duration * 17150
            )  # Speed of sound is approximately 343 meters per second
            distance = round(distance, 2)

            return distance
        except Exception as e:
            print(e)

            # return nothing, it's fine, let's try that in the next round...
            return None

    def update_sensors(self):
        distance = self.get_ultrasonic_distance()
        if distance:
            self.sensors = [distance]  # WE only got ultrasonic

    def lane_keep(self, x_ref):
        error = x_ref - const.CAR_MIDDLE_X
        current_t = time.time()

        delta_t = current_t - self.prev_t
        self._update_integral(self.integral_sum + error * delta_t)
        deriv = (error - self.prev_error) / delta_t

        new_steering = (
            const.KP * error + const.KI * self.integral_sum + const.KD * deriv
        )
        self.set_steering(int(new_steering))

        self.prev_error = error
        self.prev_t = current_t

        # print(
            # f"error: {error} | deriv: {deriv} | integral: {self.integral_sum} | steer: {new_steering}"
        # )
        # time.sleep(0.09)

    def _update_integral(self, new_val):
        self.integral_sum = new_val

        if abs(self.integral_sum) > const.WINDUP_MAX:
            self.integral_sum = (
                const.WINDUP_MAX if self.integral_sum > 0 else -const.WINDUP_MAX
            )

    def full_stop(self):
        self.set_steering(0)
        self.set_speed(0)
        time.sleep(1)

    def turn(self, speed, turning_time):
        time1 = time.time()
        while (time.time() - time1) < turning_time:
            self.set_steering(speed)
            self.set_speed(const.TURN_STATE_SPEED)

    def go_back(self, t):
        time1 = time.time()
        while (time.time() - time1) < t:
            self.set_speed(const.GO_BACK_STATE_SPEED)
        self.set_speed(0)

    def handle_turn(self, turn_side):
        time.sleep(const.TURN_WAIT_TIME)
        print(f"Turning {turn_side}.")

        if turn_side == "left":
            # turn left
            # from which lane - left or right?
            if self.current_side == "right":
                self.go_back(4.5)
                self.turn(-100, 10)
            else:
                self.turn(-80, 8)
        else:
            # turn right
            # from which lane - left or right?
            if self.current_side == "left":
                self.go_back(8)
            else:
                self.go_back(4.5)
            self.turn(100, 10)

    def handle_intersection(self, sign_state):
        # it's a turn
        time.sleep(const.INTERSECTION_WAIT_TIME)
        print(f"Turning {sign_state}.")

        if sign_state == "left":
            # which lane?
            if self.current_side == "right":
                self.turn(-45, 13)
            else:
                self.turn(-50, 12)

        elif sign_state == "straight":
            self.turn(0, 11)

        elif sign_state == "right":
            # which lane?
            if self.current_side == "right":
                self.turn(65, 9.5)
            else:
                self.turn(70, 11)

    def handle_obstacle(self):
        if self.sensors[0] < const.OBSTABLE_DETECTION_THRESHOLD:
            self.full_stop()
            print(f"Obstable detected at {self.current_side}")

            time.sleep(const.OBSTACLE_WAIT_TIME)
            if self.current_side == "right":
                self.turn(-100, 5.5)
                self.turn(100, 6.5)
                self.turn(-100, 2.5)
            else:
                self.turn(100, 4)

    def save_log_data(self):
        print(self.settings)

        if self.settings["logging"]["image"]:
            img = Image.fromarray(self.latest_img[:, :, ::-1])
            img.save(os.path.join(self.imgs_saving_path, f"camera_{time.time()}.jpg"))

        if self.settings["logging"]["state"]:
            cursor = self.db_conn.cursor()
            sensors_data = json.dumps({"ultrasonic": self.sensors})

            cursor.execute(
                "INSERT INTO car_state (speed, steering, sensors, dt) VALUES (?, ?, ?, ?)",
                (
                    self.get_speed(),
                    self.steer,
                    sensors_data,
                    dt.datetime.now(),
                ),
            )

            self.db_conn.commit()
            cursor.close()

    def sync_settings(self):
        cursor = self.db_conn.cursor()

        # log_configs
        cursor.execute("SELECT * FROM car_setting where name='log_configs';")
        log_configs_setting = cursor.fetchone()
        self.settings["logging"] = json.loads(
            log_configs_setting[2]
        )  # value is index of 2

        # movement
        cursor.execute("SELECT * FROM car_setting where name='movement';")
        movement_setting = cursor.fetchone()
        self.settings["movement"] = json.loads(
            movement_setting[2]
        )  # value is index of 2

        cursor.close()

    def send_arduino_data(self):
        data = {
        	'servo_angle': self.steer,
        	'motors_speed': abs(self.speed),
        	'is_forward': True if self.speed > 0 else False
    	}
        
    
        data_bytes = json.dumps(data).encode()
        self.arduino.write(data_bytes)
        # print(f'sent json data: {data}')
   	 
    def set_steering(self, angle):
        if angle > 0:
            if angle > const.MAX_ANGLE:
                angle = const.MAX_ANGLE

        else:
            if angle < const.MIN_ANGLE:
                angle = const.MIN_ANGLE

        self.steer = angle

    def set_speed(self, speed):
        self.speed = speed

    def get_speed(self):
        return self.speed

    def stop_moving(self):
        self.set_speed(0)
        self.set_steering(0)

    def denormalize_angle(self, normalized_angle):
        return normalized_angle * (const.MAX_ANGLE - const.MIN_ANGLE) + const.MIN_ANGLE
    
    def preprocess_img_rgb(self, image):
        image = cv2.resize(image, (128, 128))
        image = exposure.equalize_adapthist(image)

        img_min = np.min(np.min(image, axis=0), axis=0)
        img_max = np.max(np.max(image, axis=0), axis=0)

        image[:, :, 0] = image[:, :, 0] - img_min[0] / (img_max[0] - img_min[0])
        image[:, :, 1] = image[:, :, 1] - img_min[1] / (img_max[1] - img_min[1])
        image[:, :, 2] = image[:, :, 2] - img_min[2] / (img_max[2] - img_min[2])

        image = self.img_transforms(image)
        return image

    def preprocess_img_grayscale(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128))
        image = image[64:,:]
        image = self.clahe.apply(image)

        img_min = np.min(image)
        img_max = np.max(image)

        image = image - img_min / (img_max - img_min)

        image = self.img_transforms(image)
        return image

    def update_steering(self, frame):
        timings = {
            'preproc': 0.0,
            'inf': 0.0,
            'postproc': 0.0
        }
    
        t1 = time.time()
        img = self.preprocess_img_grayscale(frame)
        preproc_img = img
        timings['preproc'] = (time.time() - t1)
        

        t1 = time.time()

        img = img.to(torch.float32).unsqueeze(dim=0).to(device)
        y = self.model(img)

        timings['inf'] += (time.time() - t1)

        t1 = time.time()
        y = y.cpu().detach().numpy()[0][0]
        steer_angle = int(self.denormalize_angle(y))

        self.set_steering(steer_angle)
        timings['postproc'] += (time.time() - t1)

        return timings, preproc_img.cpu().permute(1,2,0).numpy()

    def close(self):
        self.cap.release()
        self.arduino.close()

        print('program teminated.')
