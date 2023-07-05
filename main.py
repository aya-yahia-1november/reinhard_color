import numpy as np
import cv2
import os
input_dir = "input_images/"
input_image_list = os.listdir(input_dir)

output_dir = "output_images/"

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std


template_img = cv2.imread('template_images/img_blue.webp')
template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
template_mean, template_std = get_mean_and_std(template_img)

for img in (input_image_list):
    input_img = cv2.imread(input_dir + img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)

    img_mean, img_std = get_mean_and_std(input_img)

    height, width, channel = input_img.shape
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                x = input_img[i, j, k]
                x = ((x - img_mean[k]) * (template_std[k] / img_std[k])) + template_mean[k]
                x = round(x)
                # boundary check
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                input_img[i, j, k] = x

    input_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_dir + "modified_" + img, input_img)
    cv2.imshow("f",input_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
"""
  ال target=img
  الsource=template
   l*=l(target)-mean(l(target))
   l=(std(source)/std(target))*(l*)+mean(l(source))
  
"""