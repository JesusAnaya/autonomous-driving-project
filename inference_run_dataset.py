import cv2
import numpy as np
import model
import math
import torch
import time
import scipy
from config import config
from model import NvidiaModel
from dataset_loader import get_inference_dataset


def angel_to_steer(degrees, cols, rows, smoothed_angle):
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    return mat, smoothed_angle


def main():
    dataset = get_inference_dataset()
    dataset_iterator = iter(dataset)

    model = NvidiaModel()
    model.load_state_dict(torch.load("./save/model.pt", map_location=torch.device(config.device)))
    model.to(config.device)
    model.eval()

    steering_wheel_1 = cv2.imread('./steering_wheel_tesla.jpg', 0)
    steering_wheel_2 = steering_wheel_1.copy()
    rows, cols = steering_wheel_1.shape

    smoothed_angle_1 = 1e-10
    smoothed_angle_2 = 1e-10

    while cv2.waitKey(20) != ord('q'):
        transformed_image, image, target = next(dataset_iterator)
        transformed_image = transformed_image.to(config.device)

        batch_t = torch.unsqueeze(transformed_image, 0)

        # Predictions
        with torch.no_grad():
            y_predict = model(batch_t)

        # Converting prediction to degrees
        pred_degrees = np.degrees(y_predict[0].item())
        target_degrees = np.degrees(target)

        print(f"Predicted Steering angle: {pred_degrees}")
        print(f"Steering angle: {pred_degrees} (pred)\t {target_degrees} (actual)")

        frame = np.array(image)
        cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        # and the predicted angle
        mat_1, smoothed_angle_1 = angel_to_steer(pred_degrees, cols, rows, smoothed_angle_1)
        dst_1 = cv2.warpAffine(steering_wheel_1, mat_1, (cols, rows))
        cv2.imshow("Pred steering wheel", dst_1)

        mat_2, smoothed_angle_2 = angel_to_steer(target_degrees, cols, rows, smoothed_angle_2)
        dst_2 = cv2.warpAffine(steering_wheel_2, mat_2, (cols, rows))
        cv2.imshow("Target steering wheel", dst_2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
