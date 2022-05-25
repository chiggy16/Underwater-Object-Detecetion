from detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz"

threshold = 0.5

def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(14, 14))
    for i in range(3):

        
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))


    return sceneRadiance

img = cv2.imread("Test/a.jpg")
# img = cv2.imread("Test/d.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
processed_img = RecoverCLAHE(img)
cv2.imwrite("CLAHE/a.jpg",processed_img)
# cv2.imwrite("CLAHE/d.jpg",processed_img)

classFile = "coco.names"
imagePath = "CLAHE/a.jpg"
# imagePath = "CLAHE/d.jpg"
# videoPath = "d.mp4"
videoPath = 0
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadMoadel()
detector.predictImage(imagePath,threshold)
# detector.predictVideo(videoPath,threshold)