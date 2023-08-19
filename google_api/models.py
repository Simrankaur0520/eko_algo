from django.db import models

class user_login(models.Model):
    name=models.TextField(blank=True)
    username=models.TextField(blank=True)
    email=models.TextField(blank=True)
    password=models.TextField(blank=True)
    user_id=models.TextField(blank=True)
    branches_alloted=models.TextField(blank=True)
    role=models.TextField(blank=True)

class ratings(models.Model):
    client_id=models.TextField(blank=True)
    source_name=models.TextField(blank=True)
    rating=models.TextField(blank=True)
    comments=models.TextField(blank=True)

class reviews(models.Model):
    client_id = models.CharField(max_length=50)
    name = models.CharField(max_length=200)
    review = models.TextField()
    rating = models.TextField()
    date = models.DateTimeField()
    sentiment = models.CharField(max_length=50)
    branch_id=models.TextField(blank=True)
    source=models.TextField(blank=True)

class competitor_reviews(models.Model):
    # client_id = models.CharField(max_length=50)
    name = models.CharField(max_length=200)
    review = models.TextField()
    rating = models.TextField()
    date = models.DateTimeField()
    sentiment = models.CharField(max_length=50)
    branch_id=models.TextField(blank=True)
    source=models.TextField(blank=True)
    competitor_branch_id=models.TextField(blank=True)