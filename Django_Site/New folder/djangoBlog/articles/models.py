from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class Article(models.Model):
    title = models.CharField(max_length=100)
    slug = models.SlugField()
    body = models.TextField()
    date = models.DateTimeField(auto_now_add=True)
    thumb = models.ImageField(default='default.png', blank=True)
    author = models.ForeignKey(User, default=None, on_delete=True)

    def __str__(self):
        return self.title

    def snippet(self):
        return self.body[:50] + "..."

#
# class Comments(models.Model):
#     comment = models.CharField(max_length=200)
#     date = models.DateTimeField(auto_now_add=True)
#     comment_giver = models.ForeignKey(User, default=None, on_delete=True)
#
#     def __str__(self):
#         return self.comment
