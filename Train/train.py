import ultralytics
import torch

def main():
    # Use Nvidia Cuda
    # torch.cuda.set_device(0)

    # Use pure CPU
    torch.cuda.set_device("cpu")

    model = ultralytics.YOLO("path_to_pretrained_model")
    model.train(data="data.yaml", pretrained = True, epochs=10, imgsz=640)
    model.val()

if __name__ == "__main__":
    main()
