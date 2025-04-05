from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.predict, name='predict'),
    path('result/', views.result, name='result'),  # Changed to separate view
]
