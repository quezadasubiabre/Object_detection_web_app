"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()


# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.object_detector.object_detector import object_detector

try:
	#registro
	registry = MLRegistry()
	# algoritmo machine learning
	od = object_detector()
	registry.add_algorithm(endpoint_name="detector_objetos",
	                            algorithm_object=od,
	                            algorithm_name="object detector",
	                            algorithm_status="production",
	                            algorithm_version="0.0.1",
	                            owner="CQuezadaSubiabre",
	                            algorithm_description="Algoritmo deteccion de objetos tensorflow API",
	                            algorithm_code='Code')

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))