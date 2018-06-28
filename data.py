import numpy as np
import pandas as pd
from scipy import ndimage
import json
import h5py
import keras

def preprocess_input(x):
  x /= 255.
  x -= 0.5
  x *= 2.
  return x   #归一化输入

def extract_lable(path):
    with open(path,'rb') as f:
        data=json.load(f)
        data=pd.DataFrame.from_dict(data)
        del data['image_url']
        data.sort_values(by='image_id', inplace=True)
        data = data.reset_index(drop=True)
        image_file=data['image_id']
        label= np.array( list(data['label_id'])).astype(np.int32)
        label= keras.utils.to_categorical(label, 80)   #对80个类型标签进行01二元编码
        return image_file,label   #返回图片和标签

def main():
    image_file, label = extract_lable('image/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json')
    image_path = 'image/resize_image_train/' + image_file
    for times in range(539):
        if times == 0:
            h5f = h5py.File('data/train_data.h5', 'w')
            x = h5f.create_dataset("x_train", (100, 299, 299,3),maxshape=(None, 299, 299,3),
                                         # chunks=(1, 1000, 1000),
                                         dtype=np.float32)
            y = h5f.create_dataset('y_train',(100,80),maxshape=(None,80),dtype=np.int32)   #使用h5py库读写超过内存的大型数据
			
        else:
            h5f = h5py.File('data/train_data.h5', 'a')
            x = h5f['x_train']
            y = h5f['y_train']
        # 关键：这里的h5f与dataset并不包含真正的数据，
        # 只是包含了数据的相关信息，不会占据内存空间
        # 仅当使用数组索引操作（eg. dataset[0:10]）
        # 或类方法.value（eg. dataset.value() or dataset.[()]）时数据被读入内存中
		
        image = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_path[times*100:(times+1)*100]))).astype(np.float32)
        # 调整数据预留存储空间（可以一次性调大些）
		
        image = preprocess_input(image)
        ytem = label[times*100:(times+1)*100]
        if times != 538:
            x.resize([times * 100 + 100, 299, 299,3])
            y.resize([times * 100 + 100, 80])
            # 数据被读入内存

            x[times * 100:times * 100 + 100] = image
            y[times * 100:times * 100 + 100] = ytem
            # print(sys.getsizeof(h5f))
            print('%d images are dealed with' %(times))
        else:
            x.resize([times * 100 + 79, 299, 299, 3])
            y.resize([times * 100 + 79, 80])
            # 数据被读入内存

            x[times * 100:times * 100 + 79] = image
            y[times * 100:times * 100 + 79] = ytem
            # print(sys.getsizeof(h5f))
            print('%d images are dealed with' % (times))

        h5f.close()
if __name__ == '__main__':
    main()