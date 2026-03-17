
import os
from os.path import join

# 提取各文件路径
def get_path_list(img_path,label_path,fov_path):
    tmp_list = [img_path,label_path,fov_path]
    res = []
    for i in range(len(tmp_list)):
        data_path = join("..",tmp_list[i])                         # 根目录和相对路径合并
        filename_list = os.listdir(data_path)                      # 提取目录中文件的名称，并返回一个列表
        filename_list.sort()                                       # 对这个列表进行升序排序
        res.append([join(tmp_list[i],j) for j in filename_list])   # 提取文件路径,嵌套列表
    return res

# 将各文件的路径写入txt文件
def write_path_list(name_list, save_path, file_name):
    f = open(join(save_path, file_name), 'w')     # 创建文件并打开
    for i in range(len(name_list[0])):
        f.write(str(name_list[0][i]) + " " + str(name_list[1][i]) + " " + str(name_list[2][i]) + '\n')  # 写入文件路径
    f.close()             # 关闭文件，文件无法被读写

if __name__ == "__main__":
    #------------Path of the dataset --------------------------------
    # if not os.path.exists(data_root_path): raise ValueError("data path is not exist, Please make sure your data path is correct")
    #train
    img_train = "Datasets/CHASEDB1/training/images/"                # 定义相对路径
    gt_train = "Datasets/CHASEDB1/training/1st_manual/"
    fov_train = "Datasets/CHASEDB1/training/mask/"
    img_test = "Datasets/CHASEDB1/test/images/"
    gt_test = "Datasets/CHASEDB1/test/1st_manual/"
    fov_test = "Datasets/CHASEDB1/test/mask/"
    #---------------save path-----------------------------------------
    save_path = "data_path_list/CHASEDB1/"
    if not os.path.exists(save_path):               # 判断该路径是否存在，若不存在创建该路径
        os.mkdir(save_path)
    #-----------------------------------------------------------------
    train_list = get_path_list(img_train, gt_train, fov_train)        # 提取train相关文件路径
    print('Number of train imgs:', len(train_list[0]))            # 输出训练图片数量
    write_path_list(train_list, save_path, 'train.txt')           # 把相关文件路径写入train.txt文件

    test_list = get_path_list(img_test, gt_test, fov_test)
    print('Number of test imgs:', len(test_list[0]))
    write_path_list(test_list, save_path, 'test.txt')

    print("Finish!")
