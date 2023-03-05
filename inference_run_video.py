import cv2
import numpy as np
import torch
from model import NvidiaModel
from torchvision import transforms
from config import config


transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(config.resize),
    transforms.Normalize(config.mean, config.std)
])


def angel_to_steer(degrees, cols, rows, smoothed_angle):
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    return mat, smoothed_angle


def crop_down(image):
    h = image.shape[0]
    w = image.shape[1]
    y = 130
    x = 60
    return image[60:int(y+h), int(x):int(x+(w-(x+70)))]


def main():
    model = NvidiaModel()
    model.load_state_dict(torch.load("./save/model.pt"))
    model.eval()

    steering_wheel_1 = cv2.imread('./steering_wheel_tesla.jpg', 0)
    rows, cols = steering_wheel_1.shape
    smoothed_angle_1 = 1e-10

    video_file_1 = "./videos/driving_california_1.mp4"
    video_file_2 = "./videos/driving_california_2.mp4"
    video_file_3 = "./videos/driving_california_3.mp4"

    video = cv2.VideoCapture(video_file_2)

    while cv2.waitKey(15) != ord('q'):
        success, image = video.read()
        if not success:
            print("Error reading video file")
            break

        image_cropped = crop_down(image)
        frame = transform_img(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)).double()
        batch_t = torch.unsqueeze(frame, 0)

        # Predictions
        with torch.no_grad():
            y_predict = model(batch_t)

        # Converting prediction to degrees
        pred_degrees = np.degrees(y_predict[0].item() * 2)

        print(f"Predicted Steering angle: {pred_degrees}")
        print(f"Steering angle: {pred_degrees} (pred)")
        cv2.imshow("frame", image_cropped)

        # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        # and the predicted angle
        mat_1, smoothed_angle_1 = angel_to_steer(pred_degrees, cols, rows, smoothed_angle_1)
        dst_1 = cv2.warpAffine(steering_wheel_1, mat_1, (cols, rows))
        cv2.imshow("Pred steering wheel", dst_1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
