# detector/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.detector_home, name='home'),
    path('predict/', views.predict_api, name='predict_api'),
]