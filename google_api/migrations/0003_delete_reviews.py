# Generated by Django 4.2.1 on 2023-08-12 19:09

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('google_api', '0002_reviews_user_login_branches_alloted_user_login_role_and_more'),
    ]

    operations = [
        migrations.DeleteModel(
            name='reviews',
        ),
    ]
