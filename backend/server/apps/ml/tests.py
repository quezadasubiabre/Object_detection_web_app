from django.test import TestCase

from apps.ml.object_detector.object_detector import object_detector
from apps.ml.object_detector import config


class MLTests(TestCase):
    def test_object_detector(self):
        input_data = config.PATH_IMAGE
        my_alg = object_detector()
        response = my_alg.inference(input_data)
        self.assertEqual('OK', response['status'])
  