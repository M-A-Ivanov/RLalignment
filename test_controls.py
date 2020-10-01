from simple_cofiguration import Environment
import numpy as np
import matplotlib.pyplot as plt
from tools import rawToImage
import os
from check_resolution import ImageEvaluation
from LEEMcontrol import oLeem

def test_image_creation():
    env = Environment(120, 20, -20)
    n_images = 5
    env.setFluffy()
    template = "state: {}, reward: {}"
    for i in range(n_images):
        state, reward, image, done = env.step(np.random.randint(0, env.n_actions))
        print(template.format(state, reward))
        if image is not None:
            plt.imshow(image)
            plt.savefig(env.path + "read_img_" + str(i) + ".png")
            plt.close()
        # success


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

def test_LEEM2000_controls():
    LEEM = oLeem(port=5568)
    LEEM.connect()
    LEEM.testConnect()
    for i in oLeem.Modules.values():
        print(i)
    for i in oLeem.Mnemonic.values():
        print(i)


if __name__ == '__main__':
    test_resolution_score()