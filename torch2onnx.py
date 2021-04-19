import torch
from torchvision import transforms
import cv2
import onnxruntime

from retinanet import model
from train import remove_prefix


def torch2onnx(model, save_path):
    """
    :param model:
    :param save_path:  XXX/XXX.onnx
    :return:
    """
    model.eval()
    data = torch.rand(1, 3, 224, 224)
    data = data.cuda()
    input_names = ["input"]
    output_names = ["out"]
    torch.onnx._export(model, data, save_path, verbose=True, export_params=True, opset_version=11,
                       input_names=input_names, output_names=output_names)
    print("torch2onnx finished.")


def torch2onnx_dynamic(model, save_path):
    """
    :param model:
    :param save_path:  XXX/XXX.onnx
    :return:
    """
    model.eval()
    data = torch.rand(1, 3, 800, 800)
    input_names = ["input"]  # ncnn需要
    output_names = ["out"]  # ncnn需要
    torch.onnx._export(model, data, save_path, verbose=True, export_params=True, opset_version=11,
                       input_names=input_names, output_names=output_names,
                       dynamic_axes={'input': [2, 3], 'out': [2, 3]})
    print("torch2onnx finished.")


if __name__ == '__main__':

    # generate onnx model
    checkpoint = torch.load('D:/program file/condaProjects/.datasets/maskDetection/weights/csv_retinanet50_80epochs.pt')
    retinanet = model.resnet50(num_classes=2)
    pretrained_dict = remove_prefix(checkpoint['model_state_dict'], 'module.')
    retinanet.load_state_dict(pretrained_dict)
    retinanet = retinanet.cuda()

    torch2onnx(retinanet, 'retinanet_maskDet.onnx')

    # running with onnx model
    # session = onnxruntime.InferenceSession('retinanet_maskDet.onnx')
    #
    # img = cv2.imread('img_path')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # tensor = transforms.ToTensor()(img)
    # tensor = tensor.unsqueeze_(0)
    #
    # result = session.run([], {"input": tensor.cpu().numpy()})
