import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
#import learning_sample

''' 샘플링한 얼굴 사진이 들어있는 경로 '''
data_path = 'faces/'

''' 샘플 파일만 리스트로 가져오기 '''
sample_files = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []

''' Training_Data에 샘플 사진 넣기'''
for i, files in enumerate(sample_files):
    image_path = data_path + sample_files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue
    ''' uint8 형식으로 배열 저장 '''
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(np.asarray(i, dtype=np.int32))

if len(Labels) == 0:
    print("There is no data to train.")
    exit()
''' LBPHFaceRecongizer 학습 '''
recognition_lbph = cv2.face.LBPHFaceRecognizer_create()
recognition_lbph.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete.")

''' 모자이크 기능 시작 '''
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

''' YOLO 학습된 객체 불러오기 '''
class_name = []
with open('classes-custom.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet('yolov4-tiny-custom.weights', 'yolov4-tiny-custom.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

detection_yolo = cv2.dnn_DetectionModel(net)
detection_yolo.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


def face_detector(img, size = 0.5):
    '''
    YOLO로 검출된 얼굴을 LBPH에 입력으로 쓰고 본인인지 확인한다.
    본인이 아닌 걸로 결과가 나오면 모자이크 처리한다.
    '''
    classes, scores, boxes = detection_yolo.detect(img, Conf_threshold, NMS_threshold)
    
    if len(boxes) == 0:
        cv2.imshow('mosaic_cam', img)
        return

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        cv2.rectangle(img, box, color, 1)

        roi = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]

        try:
            roi = cv2.resize(roi, (200, 200))
        except:
            break

        ''' 검출된 사진을 흑백으로 변환 '''
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ''' 위에서 학습한 모델로 예측 시도 '''
        result = recognition_lbph.predict(roi)

        ''' result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다. '''
        # 500?
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'%'

        cv2.putText(img,display_string,(box[0],box[1]), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        ''' 80%이하면 다른 사람으로 간주'''
        if confidence >= 77:
            cv2.putText(img, "It's you!", (box[0], box[1]+box[3]+19), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        else:
            roi = img[box[1]:box[1]+box[3], box[0]:box[0]+box[1]]

            try:
                roi = cv2.resize(roi,(box[2]//15,box[3]//15))
                roi = cv2.resize(roi,(box[2],box[3]), interpolation=cv2.INTER_AREA)
            except:
                break

            img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = roi
            cv2.putText(img, "who?", (box[0], box[1]+box[3]+19), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('mosaic_cam',img)
    
''' 카메라 열기 '''
cap = cv2.VideoCapture(0)

while True:
    ''' 카메라로 부터 사진 한장 읽기 '''
    ret, frame = cap.read()
    if ret ==False:
        break
    ''' 얼굴 검출 시도 '''
    face_detector(frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()