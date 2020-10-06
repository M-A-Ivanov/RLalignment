import numpy as np
import matplotlib.pyplot as plt
from tools import rawToImage
import os
from check_resolution import ImageEvaluation
from LEEMcontrol import oLeem
from UVIEWcontrol import oUview
from myLEEM import LEEM_remote

def test_resolution_score():
    template = " : resolution_score is {}"
    ResChecker = ImageEvaluation()
    path = "C:\\Users\\User\\OneDrive - Cardiff University\\Data\\LEEM\\for testing\\alignment"
    file_list = os.listdir(path)
    for i, file in enumerate(file_list):
        if not file.endswith('.dat'):
            continue
        fullpath = path + "\\" + file
        image = rawToImage(fullpath)[150:-150, 150:-150]
        ResChecker.inputImage(image)
        laplacian = ResChecker.getLaplacian()
        plt.subplot(2, 1, 1), plt.imshow(image, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 1, 2), plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.show()
        print(file + template.format(ResChecker.scoreFocus()))


def test_object_detection():
    ResChecker = ImageEvaluation()
    path = "C:\\Users\\User\\OneDrive - Cardiff University\\Data\\LEEM\\for testing\\alignment"
    file_list = os.listdir(path)
    for i, file in enumerate(file_list):
        if not file.endswith('.dat'):
            continue
        fullpath = path + "\\" + file
        image = rawToImage(fullpath)[150:-150, 150:-150]
        ResChecker.inputImage(image)
        objects = ResChecker.detectObjects()
        plt.imshow(objects)
        plt.show()


def test_LEEM_control(change_module=False):
    LEEM = oLeem(port=5566, ip='localhost')
    LEEM.connect()
    LEEM.testConnect()
    for i in LEEM.Modules.values():
        print(i)
    for i in LEEM.Mnemonic.values():
        print(i)
    # if change_module:
    #     current = LEEM.getValue('Cond. Lens 3')
    #     print('The value of CL3 is: {}'.format(current))
    #     current = current + (1/40)*current
    #     print('Changing by +2.5% ... ')
    #     LEEM.setValue('Cond. Lens 3', current)
    #     current = LEEM.getValue('Cond. Lens 3')
    #     print('New current is: {}'.format(current))

    LEEM.disconnect()


def test_Uview_control():
    Uv = oUview(port=5570, ip='localhost')  # set port
    img = Uv.getImage()
    print('shape: {} and type: {}'.format(img.shape, type(img)))
    plt.imshow(img)
    plt.show()
    img2 = Uv.getImage()
    plt.imshow(img2)
    plt.show()
    Uv.disconnect()


def test_LEEM_modules():
    LEEM = LEEM_remote()
    LEEM.print_state(save=True)
    print(LEEM.change)


if __name__ == '__main__':
    test_LEEM_modules()