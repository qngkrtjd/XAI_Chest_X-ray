"""

test_image 폴더 안에 있는 이미지 파일들에 대한 정보를 BBox_List_2017.csv에서 검색해 따로 저장.

"""

import os
import csv

# 이미지 파일 경로
image_folder = 'test_image/'

# CSV 파일 경로
csv_file = 'BBox_List_2017.csv'

# 저장할 CSV 파일 경로
output_csv_file = 'test_image_info.csv'

# 이미지 파일 목록 가져오기
image_files = os.listdir(image_folder)

# 중복을 제거한 이미지 파일 목록
unique_image_files = set(image_files)

# 이미지 파일 이름을 기반으로 딕셔너리 생성
image_info = {image_file: [] for image_file in unique_image_files}

# CSV 파일 읽기
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 라인 스킵

    # CSV 파일 내 각 라인을 읽어서 이미지 정보 추출
    for line in reader:
        image_index = line[0]
        finding_label = line[1]
        x = float(line[2])
        y = float(line[3])
        w = float(line[4])
        h = float(line[5])

        # 이미지 파일 이름 추출
        image_file = image_index

        # 이미지 파일이 test_image 폴더에 존재하는 경우에만 정보 추가
        if image_file in unique_image_files:
            image_info[image_file].append((finding_label, x, y, w, h))

# 추출한 이미지 정보를 CSV 파일로 저장
with open(output_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    # 헤더 작성
    writer.writerow(['Image File', 'Finding Label', 'Bbox[x', 'y', 'w', 'h]'])

    # 이미지 정보 작성
    for image_file, info_list in image_info.items():
        for finding_label, x, y, w, h in info_list:
            writer.writerow([image_file, finding_label, x, y, w, h])

print(f"이미지 정보가 {output_csv_file} 파일로 저장되었습니다.")
