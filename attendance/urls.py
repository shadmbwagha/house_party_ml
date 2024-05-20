from django.urls import path
from . import views
from django.urls import include, path

urlpatterns = [
    path('predict/', views.predict_attendance, name='predict_attendance'),

]
