from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/tiny-yolo-voc.cfg", "train": 1,"dataset":"VOCdevkit/VOC2007/JPEGImages",
           "annotation":"VOCdevkit/VOC2007/Annotations","load":"bin/tiny-yolo-voc.weights"}

tfnet = TFNet(options)
tfnet.train()

imgcv = cv2.imread("./sample_img/dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)