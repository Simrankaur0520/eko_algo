# Generated by Django 4.2.1 on 2023-08-12 21:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('google_api', '0006_reviews'),
    ]

    operations = [
        migrations.CreateModel(
            name='ratings',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source_name', models.TextField(blank=True)),
                ('rating', models.TextField(blank=True)),
                ('comments', models.TextField(blank=True)),
            ],
        ),
    ]
