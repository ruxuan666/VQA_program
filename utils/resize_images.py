# 预处理输入图片
import os
import argparse
from PIL import Image


#将image,resize成size型号
def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)


def resize_images(input_dir, output_dir, size):
    #输入路径为./datasets/Images:包含三个文件夹，-训练集（82783），-验证集（40504），-测试集(81434)，每个文件夹下是图片
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir):#浏览该目录下的子目录
        if not idir.is_dir(): #若该目录不存在
            continue
        if not os.path.exists(output_dir+'/'+idir.name):
            os.makedirs(output_dir+'/'+idir.name)  #建立输出目录
        images = os.listdir(idir.path)#该文件夹下的文件（图片名）
        n_images = len(images)
        for iimage, image in enumerate(images):
            try:
                with open(os.path.join(idir.path, image), 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img.save(os.path.join(output_dir+'/'+idir.name, image), img.format)
            except(IOError, SyntaxError) as e:
                pass
            if (iimage+1) % 1000 == 0:
                print("[{}/{}] Resized the images and saved into '{}'."
                      .format(iimage+1, n_images, output_dir+'/'+idir.name))

#转换没有转换完成的数据
def resize_images1(output_dir='../datasets/Resized_Images',size=[224,224]):
    idir='../datasets/Images/val2014' #返回上一级目录下的datasets
    images = os.listdir(idir)  # 该文件夹下的文件（图片名）
    n_images = len(images)#图片数目
    if not os.path.exists(output_dir + '/val2014'):
        os.makedirs(output_dir + '/val2014')  # 建立输出目录
    for iimage, image in enumerate(images):
        if iimage<40319: continue #已经保存了40319张图片
        try:
            with open(os.path.join(idir, image), 'r+b') as f:
                with Image.open(f) as img:
                    img = resize_image(img, size)
                    img.save(os.path.join(output_dir + '/val2014' , image), img.format)
        except(IOError, SyntaxError) as e:
            pass
        if (iimage + 1) % 1000 == 0:
            print("[{}/{}] Resized the images and saved into '{}'."
                  .format(iimage + 1, n_images, output_dir + '/val2014'))


def main(args):

    input_dir = args.input_dir #赋相应名字下的设定值
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(input_dir, output_dir, image_size)

    
if __name__ == '__main__':

    """parser = argparse.ArgumentParser()
    #给定输入参数默认值
    parser.add_argument('--input_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA/Images',
                        help='directory for input images (unresized images)')

    parser.add_argument('--output_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA/Resized_Images',
                        help='directory for output images (resized images)')

    parser.add_argument('--image_size', type=int, default=224,
                        help='size of images after resizing')

    args = parser.parse_args()

    main(args)"""
    images = os.listdir('../datasets/Resized_Images/val2014')  # 该文件夹下的文件（图片名）
    n_images = len(set(images)) #val:40319;test:29862
    print(n_images)
    #resize_images1()
