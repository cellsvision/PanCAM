
import os
import random
import copy

def split_data_list(input_file_list, val_ratio):
    tmp_file_list = copy.deepcopy(input_file_list)


    total_num = len(tmp_file_list)

    val_num = int(total_num * val_ratio)    

    print(val_num)

    random.shuffle(tmp_file_list)

    tmp_train_list = tmp_file_list[val_num:]
    tmp_val_list = tmp_file_list[:val_num]

    return tmp_train_list, tmp_val_list



input_whole_file_path = 'path_to/dataset/labeled.txt'
input_detail_dir = 'path_to/dataset/details_info'


output_train_file_path = 'path_to/dataset/train.txt'
output_val_file_path = 'path_to/dataset/val.txt'


val_ratio = 0.2


whole_file_list = []
with open(input_whole_file_path) as f:
    for line in f.readlines():
        whole_file_list.append(line.rstrip())

print(len(whole_file_list))



intput_fg_dir = os.path.join(input_detail_dir, 'fg')
input_bg_dir = os.path.join(input_detail_dir, 'no_fg')

fg_file_list = [os.path.splitext(x)[0] for x in os.listdir(intput_fg_dir) if x.endswith(".txt")]
bg_file_list = [os.path.splitext(x)[0] for x in os.listdir(input_bg_dir) if x.endswith(".txt")]


join_file_list = []
fg_only_file_list = []
bg_only_file_list = []


for f in fg_file_list:
    if f in bg_file_list:
        join_file_list.append(f)
    else:
        fg_only_file_list.append(f)

for f in bg_file_list:
    if f  not in fg_file_list:
        bg_only_file_list.append(f)        

print(len(join_file_list))
print(len(fg_only_file_list))
print(len(bg_only_file_list))

train_list = []
val_list = []

tmp_train_list, tmp_val_list = split_data_list(join_file_list, val_ratio)
train_list.extend(tmp_train_list)
val_list.extend(tmp_val_list)

tmp_train_list, tmp_val_list = split_data_list(fg_only_file_list, val_ratio)
train_list.extend(tmp_train_list)
val_list.extend(tmp_val_list)

tmp_train_list, tmp_val_list = split_data_list(bg_only_file_list, val_ratio)
train_list.extend(tmp_train_list)
val_list.extend(tmp_val_list)

print(len(train_list))
print(len(val_list))


with open(output_train_file_path, 'w') as f:
    for tf in train_list:
        f.write(tf + '\n')

with open(output_val_file_path, 'w') as f:
    for tf in val_list:
        f.write(tf + '\n')
