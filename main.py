from src import utils, const, Car
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt


def show_frame(frame):
    text = str(car.steer)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Set the text position
    text_x = 10  # 10 pixels from the right edge
    text_y = 20  # 20 pixels from the top edge

    # Draw the text on the image
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


    # cv2.imshow("Car frame", frame)
    cv2.imwrite(f'./imgs/test/img_{int(time.time())}.png', frame)
    
    key = cv2.waitKey(1)

    if key == ord("w"):
        frame = car.get_image()
        cv2.imwrite(
            os.path.join(
                os.getenv("IMG_SAVING_PATH"), f"frame_{time.time():.5f}.jpg"
            ),
            frame,
        )



np.seterr(all="ignore")

times = {
    'ignite': [],
    'arduino': [],
    'cam': [],
    'e2e': {
        'preproc': [],
        'inf': [],
        'postproc': []
    },
    'showing_img': [],
    'sensors': []
}


car = Car(os.getenv("IMG_SAVING_PATH"))
is_car_stopped = False

try:
    car.sync_settings()
    
    t1 = time.time()
    car.ignite()
    times['ignite'].append(time.time() - t1)
    
    while True:
        ## getting data
        car.sync_settings()
        
        t1 = time.time()
        car.update_sensors()
        t2 = time.time()
        times['sensors'].append(t2 - t1)

        t1 = time.time()
        frame = car.get_image()
        t2 = time.time()
        times['cam'].append(t2-t1)

        car.log_server_counter += 1
        if car.log_server_counter >= const.LOG_SERVER_EVERY:
            print("logging")

            car.save_log_data()
            car.log_server_counter = 0

        if car.settings["movement"]["stop"]:
            car.stop_moving()
            print("car is stopped")
            car.send_arduino_data()
            is_car_stopped = True
            continue


        if is_car_stopped:
            is_car_stopped = False

            car.update_steering(frame)
            car.ignite()

        
        steer_times, preproc_img = car.update_steering(frame)
        times['e2e']['preproc'].append(steer_times['preproc'])
        times['e2e']['inf'].append(steer_times['inf'])
        times['e2e']['postproc'].append(steer_times['postproc'])


        # preproc_img = car.preprocess_img_grayscale(frame)
        # preproc_img = preproc_img.cpu().squeeze(dim=0).numpy()
        # print(f'preproc_img.shape: {preproc_img.shape}')


        t1 = time.time()
        car.set_speed(const.NORMAL_STATE_SPEED)
        
        car.send_arduino_data()
        time.sleep(10e-3) # small delay for waiting for motors full response
        t2 = time.time()
        times['arduino'].append(t2-t1)

        # print(f'speed: {car.speed} | steer: {car.steer}')


        # show car + masks

        # Get the text size
        # t1 = time.time()
        # show_frame(frame)
        # plt.gray()
        # import numpy as np
        # plt.imshow(np.reshape(preproc_img, (64, 128)))
        # plt.savefig(f'./imgs/test/img_{int(time.time())}.png')
        # t2 = time.time()
        # times['showing_img'].append(t2-t1)

        # time.sleep(0.05)
except KeyboardInterrupt:
    print('keyboard interrupt. stopping the car')

finally:
    car.stop_moving()
    car.send_arduino_data()
    car.close()

    print('\ntimings\n')
    for key, times_arr in times.items():
        if key == 'e2e' or len(times_arr) == 0:
            continue

        print(f'{key} -> {sum(times_arr) / len(times_arr)}s')
    
    print(f"e2e - preproc: {sum(times['e2e']['preproc']) / len(times['e2e']['preproc'])}")
    print(f"e2e - inf: {sum(times['e2e']['inf']) / len(times['e2e']['inf'])}")
    print(f"e2e - postproc: {sum(times['e2e']['postproc']) / len(times['e2e']['postproc'])}")

