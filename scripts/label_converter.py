## camera : 0 phone :2
import numpy as np
import glob
## 1 -> 2

def process_file(input_filename, output_filename):
    lines_to_keep = []
    with open(input_filename, 'r') as file:
        for line in file:
            class_label, *values = line.split()
            class_label = int(class_label)

            # 클래스가 0인 경우 또는 클래스가 2인 경우를 1로 변경하여 lines_to_keep 리스트에 추가
            if class_label == 0:
                lines_to_keep.append('0 ' + ' '.join(values) + '\n')
            elif class_label == 1:
                lines_to_keep.append('2 ' + ' '.join(values) + '\n')
            # 클래스가 1 또는 3인 경우에는 건너뜀
            # elif class_label == 2 or class_label == 3:
                # continue
            else : 
                print('what?')

    # 수정된 내용을 새로운 파일로 저장
    with open(output_filename, 'w') as file:
        file.writelines(lines_to_keep)

# 여러 파일에 대해 작업을 반복하려면 파일 이름을 반복문으로 지정
# 예를 들어, 파일명이 data1.txt, data2.txt, data3.txt인 경우:
# file_names = ['data1.txt', 'data2.txt', 'data3.txt']
file_names = glob.glob('./data/save_img_copy/**/**.txt',recursive=True)
print(len(file_names))

for input_file in file_names:
    output_file = input_file  # 수정된 파일명을 동일하게 유지하려면 input_file과 동일하게 설정
    process_file(input_file, output_file)
