# Generated by Django 5.1.3 on 2024-12-21 17:19

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="LungCancerTrackerModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("age", models.IntegerField()),
                (
                    "gender",
                    models.CharField(
                        choices=[(1, "Male"), (0, "Female")], max_length=1
                    ),
                ),
                ("smoking", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                (
                    "yellow_fingers",
                    models.IntegerField(choices=[(1, "YES"), (0, "NO")]),
                ),
                ("anxiety", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                ("peer_pressure", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                (
                    "chronic_disease",
                    models.IntegerField(choices=[(1, "YES"), (0, "NO")]),
                ),
                ("fatigue", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                ("allergy", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                ("wheezing", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                ("alcohol", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                ("coughing", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                (
                    "shortness_of_breath",
                    models.IntegerField(choices=[(1, "YES"), (0, "NO")]),
                ),
                (
                    "swallowing_difficulty",
                    models.IntegerField(choices=[(1, "YES"), (0, "NO")]),
                ),
                ("chest_pain", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
                ("lung_cancer", models.IntegerField(choices=[(1, "YES"), (0, "NO")])),
            ],
        ),
    ]
