from django.urls import path
from google_api.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('login',login),
    path('test_call',test_call),
    path('store_data',store_data),
    path('competitor_store_data',competitor_store_data),
    path('google_dashboard',google_dashboard),

    path('dashboard',dashboard),
    # path('rating_over_time',rating_over_time),


]
