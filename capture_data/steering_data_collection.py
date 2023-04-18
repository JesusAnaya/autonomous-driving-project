import pandas as pd
import os
import math


def get_vehicle_speed(vehicle):
    velocity = vehicle.get_velocity()
    return float(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))


class SteeringDataCollector(object):
    def __init__(self, word, vehicle, camera, root_dir=".", csv_file_name="steering_data.csv", output_dir="images"):
        self.word = word
        self.vehicle = vehicle
        self.camera = camera
        self.capture_started = False
        self.output_dir = os.path.join(root_dir, output_dir)
        self.csv_file_name = os.path.join(root_dir, csv_file_name)

        # Load data if the file exists, if not create a new one
        if os.path.exists(self.csv_file_name):
            self.csv_file = pd.read_csv(self.csv_file_name)
        else:
            self.csv_file = pd.DataFrame(columns=["frame_name", "steering_angle"])

        # check if the directory exists
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def start_capture(self):
        self.capture_started = True

    def stop_capture(self):
        self.capture_started = False
        self.csv_file.to_csv(self.csv_file_name, index=False)

    def run_capture(self, image, vehicle):
        if not self.capture_started:
            return

        # Capture every other frame
        if int(image.frame) % 2 != 0:
            return

        # Only capture data when the vehicle is moving
        if get_vehicle_speed(vehicle) < 5.0:
            return

        steering_angle = vehicle.get_control().steer
        frame_name = f"{image.frame}.jpg"
        fame_path = os.path.join(self.output_dir, frame_name)

        self.csv_file.loc[len(self.csv_file.index)] = [frame_name, f"{steering_angle:.8f}"]

        image.save_to_disk(fame_path)
        self.csv_file.to_csv(self.csv_file_name, index=False)
