import os
from shutil import copyfile as copyfile
from PIL import Image, ImageEnhance
import os
from IPython.display import display

def enhance_contrast(image, factor):
    """ Changes the contrast of the image
    
        :param Image image: image to be enhanced
        :param double factor: enhancement factor

        :returns: Enhanced image

        :rtype: Image
    """
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(factor)
    return image_enhanced

def enhance_brightness(image, factor):
    """ Changes the brightness of the image
    
        :param Image image: image to be enhanced
        :param double factor: enhancement factor

        :returns: Enhanced image

        :rtype: Image
    """
    enhancer = ImageEnhance.Brightness(image)
    image_enhanced = enhancer.enhance(factor)
    return image_enhanced

def enhance_sharpness(image, factor):
    """ Changes the sharpness of the image
    
        :param Image image: image to be enhanced
        :param double factor: enhancement factor

        :returns: Enhanced image

        :rtype: Image
    """
    enhancer = ImageEnhance.Sharpness(image)
    image_enhanced = enhancer.enhance(factor)
    return image_enhanced

def save_image(image, filepath, filename):
    """ Saves the image
    
        :param Image image: image to be saved
        :param string filepath: path to the folder where the image is to be saved
        :param string filename: file name of the image
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    image.save(filepath+filename+".jpg")
    
def save_gt(gt, filepath, filename):
    """ Saves the ground truth density map
    
        :param npy gt: ground truth density map to be saved
        :param string filepath: path to the folder where the density map is to be saved
        :param string filename: file name of the density map
    """
    copyfile(gt, filepath+filename+".npy")
