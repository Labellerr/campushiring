from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from tracker.views import video_upload_and_process

urlpatterns = [
    path('', video_upload_and_process, name='video_upload'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
