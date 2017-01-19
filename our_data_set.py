from PIL import Image
import numpy as np
import glob
event_png_pair = []
for file in glob.glob('image_set/image1/image1_150.jpg'):
    img = Image.open(file, 'r')
    ''' Some image resizing code '''
    img_conv = img.convert("L")
    datum = np.array(img_conv)
    ''' Some name parsing below '''
    name = file
    name = name.replace('.png', '')[::-1]
    name_list = list(name)
    number_char_list = name_list[:name_list.index('_')]
    number_list = number_char_list[::-1]
    event_number = int(''.join(number_list))
    ''' Create tuple with event number and corresponding np array from image '''
    event_png_pair.append((event_number, datum))
