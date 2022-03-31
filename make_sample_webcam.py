import cv2

''' A threshold used to filter boxes by confidences.
    A threshold used in non maximum suppression '''
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

''' 1.Person 2.Knife 3.Cigarette 분류할 데이터 종류를 나열한 클래스 파일 '''
class_name = []
with open('classes-custom.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

''' 미리 학습된 yolo wrights와 설정파일인 cfg를 주어서 yolo deeplearning network 객체 생성 '''
net = cv2.dnn.readNet('yolov4-tiny-custom.weights', 'yolov4-tiny-custom.cfg')

''' (YOLO) GPU를 사용할 수 있게 설정 -> GPU가 없거나 돌릴수 없으면 자동으로 CPU로 실행 '''
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

''' deeplearning network를 받아 탐지 모델을 생성 '''
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


def face_extractor(img):
    '''
    lbph 알고리즘의 학습 입력에 맞게 사진을 변환하는 함수
    model.detect : 얼굴찾기(YOLO) / frame을 입력으로 탐지된 결과 반환
    탐지된 얼굴 중에서 얼굴이 없으면 len(boxes)==0 NONE을 리턴하고 얼굴이 있으면 얼굴부분의 화면의 좌표만 잘라서 반환해준다.
    '''

    if img is not None:
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

        if len(boxes) == 0:
            return None
        for (classid, score, box) in zip(classes, scores, boxes):
            cropped_face = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        return cropped_face

    else:
        return None


''' 카메라 실행 -> 인자 0 / 서버에서 영상을 받아오려면 -> 인자 URL '''
cap = cv2.VideoCapture(0)

''' 저장할 이미지 카운트 변수 '''
count = 0

''' 얼굴 샘플 추출 시작 '''
while True:
    ''' 영상으로부터 사진 1장 얻기 '''
    ret, frame = cap.read()

    ''' 
    face_extractor 얼굴 감지 함수 -> 27번째줄 정의되어있음 
    얼굴 이미지 크기를 200x200으로 조정 
    크기 조정된 이미지를 흑백으로 변환
    path 위치에 사진 저장 
    
    화면에 얼굴과 현재 저장 개수 표시
    50개 데이터를 추출하도록 설정되어있음 
    file_path 수정해서 사용
    '''
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = './faces' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1) == 13 or count == 10:
        break

cap.release()
cv2.destroyAllWindows()

print('Collecting Samples Complete')
