from PIL import Image,ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

#def resize_image(image):
    #image = image.resize([299, 299], Image.ANTIALIAS)
    #return image

def main():
    for split in ['train','validation']:
        folder='image/ai_challenger_scene_%s_20170904/scene_%s_images_20170904'%(split,split)
        resized_folder='image/resize_image_%s'%split
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        print('Start resizing %s images.'%split )
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:   #读取二进制文件
                with Image.open(f) as image:
                    image = image.resize([299, 299], Image.ANTIALIAS)   #将图片大小压缩为Xception标准输入299x299
                    image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print('Resized %s images: %d/%d' % (split, i, num_images))


if __name__ == '__main__':
    main()