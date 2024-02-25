import torch
import torch.nn as nn
from torchvision import transforms
import argparse
from torchvision.models import vgg16, vgg19, resnet50, densenet121
from AlexLikeNet import AlexLikeNet
from RestNetLikeNet import ResNet18LikeNet
import cv2
import os

def choose_model(model_type, class_count):
    if model_type == 'VGG16':
        model = vgg16(pretrained=False, num_classes=class_count)
    elif model_type == 'VGG19':
        model = vgg19(pretrained=False, num_classes=class_count)
    elif model_type == 'ResNet50':
        model = resnet50(pretrained=False, num_classes=class_count)
    elif model_type == 'DenseNet121':
        model = densenet121(pretrained=False, num_classes=class_count)
    elif model_type == 'custom1':
        model = AlexLikeNet(class_count)
    else:
        model = ResNet18LikeNet(class_count)
    # Add softmax layer to the end of the model
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    return model

parser = argparse.ArgumentParser(description='模型資料。')
parser.add_argument(
    '--camera-source',
    help='攝影機編號',
    type=int,
    default=0,
)
parser.add_argument(
    '--weights-file',
    # required=True,
    help='模型參數檔案',
    default='model.pth',
)
parser.add_argument(
    '--input-width',
    type=int,
    default=256,
    help='模型輸入寬度',
)
parser.add_argument(
    '--input-height',
    type=int,
    default=256,
    help='模型輸入高度',
)
parser.add_argument(
    '--model-type',
    choices=('VGG16', 'VGG19', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'custom1', 'custom2'),
    default='custom2',
    help='選擇模型類別',
)
# 參數設定
args = parser.parse_args()
camera_src = args.camera_source     # 選定攝影機
weights_file = args.weights_file
num_classes = 2
# Model selection
model = choose_model(args.model_type, class_count=num_classes)
print(model)

model.load_state_dict(torch.load(weights_file))
model.eval()

# 開啟相機
cap = cv2.VideoCapture(camera_src + cv2.CAP_DSHOW)
if cap.isOpened() is not True:
    print('Camera is not opened.')
    exit()  # 強制結束 python 程式 (因為開啟相機失敗)

window_title = 'Camera: {}'.format(camera_src)
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.input_height, args.input_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ret = True
while ret:
    ret, frame = cap.read()

    if ret:
        # Preprocess the image
        with torch.no_grad():
            X = transform(frame)
            X = X.unsqueeze(0)
            pred = model(X)
            if pred.argmax(1) == 1:
                print('I see bee.')
            else:
                print('I see ant.')

        key = cv2.waitKey(5) & 0xFF
        if key == 27: # 判斷輸入鍵是否為 ESC
            print('ESC is pressed by user.')
            break

        cv2.imshow(window_title, cv2.flip(frame, 1))

# 終止影像裝置
cap.release()               # 釋放 video capture 資源
cv2.destroyAllWindows()     # 關閉所有視窗