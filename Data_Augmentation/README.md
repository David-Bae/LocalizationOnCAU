Data Augmentation 방법

1. 'data/original_data' 폴더에 증강하고 싶은 이미지들을 저정합니다.

2. 'data_augmentation.py'를 실행합니다. 'data_augmentation.py'파일에서는 'mytransform.py'파일에 구현된 transform 함수들을 호출하여 이미지를 변환하고 변환된 이미지를 'data/transformed_data' 폴더에 저장합니다.

3. 'data/transformed_data' 폴더에 6장의 변환된 이미지가 저장됩니다.

주의 : 라이브러리를 사용하지 않고 transform 함수를 직접 구현하였기 때문에 속도가 매우 느립니다. 제 노트북 기준 1이미지당 최대 2분 30초 정도가 걸렸습니다.