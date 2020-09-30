import cv2 as cv
import cvlib
from cvlib.object_detection import draw_bbox


class ImageEvaluation(object):
    def __init__(self):
        """":param focus_score: measure defocus
            :param uniform_focus_score: focus uniformity
            :param astigm_score: measure astigmatism
            :param uniform_intensity_score: measure intensity unifromness
            """
        self.image = None
        self.focus_score = None
        self.astigm_score = None
        self.uniform_intensity_score = None

    def inputImage(self, inputs):
        self.image = inputs

    def scoreFocus(self):
        edges = self.getLaplacian()
        _, std = cv.meanStdDev(edges)
        return abs(std)

    def detectObjects(self):
        bbox, label, conf = cvlib.detect_common_objects(self.image)
        output_image = draw_bbox(self.image, bbox, label, conf)

        return output_image

    def getLaplacian(self):
        return cv.Laplacian(self.image, ddepth=cv.CV_64F, ksize=11)
