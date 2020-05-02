import inspect
from apps.ml.registry import MLRegistry

from django.test import TestCase

from apps.ml.object_detector.object_detector import object_detector
from apps.ml.object_detector import config


class MLTests(TestCase):
    def test_object_detector(self):
        input_data = config.PATH_IMAGE
        my_alg = object_detector()
        response = my_alg.inference(input_data)
        self.assertEqual('OK', response['status'])
  

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "object_detector"
        algorithm_object = object_detector()
        algorithm_name = "object detector"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "CQuezadaSubiabre"
        algorithm_description = "Object detector tensorflow API"
        algorithm_code = inspect.getsource(object_detector)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)