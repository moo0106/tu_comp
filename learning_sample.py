"""
특정 인물의 얼굴을 알아보기 위해
추출한 샘플을 사용하여 LBPH 학습을 진행한다.
"""
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

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
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete.")