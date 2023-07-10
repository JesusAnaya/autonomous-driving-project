import torch
from model import NvidiaModel


def main():
    # Ensure model and input are on the same device
    device = torch.device('cpu')

    # Load your pretrained PyTorch model here
    model = NvidiaModel()
    model.load_state_dict(torch.load("./save/model.pt", map_location=device))
    model.to(device)  # Move the model to the device
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input that matches the input format of the model
    # The input format of the model is (batch_size, channels, height, width)
    # for this case, width = 200, height = 66, channels = 3
    dummy_input = torch.randint(0, 255, (1, 3, 66, 200), dtype=torch.float32, device=device) / 255.0

    # Export the model to an ONNX file
    torch.onnx.export(
        model,
        dummy_input,
        './save/drive_net_model.onnx',
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
    )

    print('Model exported to "drive_net_model.onnx".')


if __name__ == '__main__':
    main()
