# Generated by Django 4.2.1 on 2023-08-13 16:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('google_api', '0010_delete_competitor_reviews'),
    ]

    operations = [
        migrations.CreateModel(
            name='competitor_reviews',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('review', models.TextField()),
                ('rating', models.TextField()),
                ('date', models.DateTimeField()),
                ('sentiment', models.CharField(max_length=50)),
                ('branch_id', models.TextField(blank=True)),
                ('source', models.TextField(blank=True)),
                ('competitor_branch_id', models.TextField(blank=True)),
            ],
        ),
    ]
