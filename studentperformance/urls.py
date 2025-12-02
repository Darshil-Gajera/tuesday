from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.upload_file, name='upload_file'),
    path('predict/', views.predict_student, name='predict_student'),
]
