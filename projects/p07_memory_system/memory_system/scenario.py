"""Scripted multi-session conversation with planted facts, updates and distractors."""
from __future__ import annotations

from dataclasses import dataclass

# (day offset, [(role, text), ...]) — one entry per session.
SESSIONS: list[tuple[float, list[tuple[str, str]]]] = [
    (0, [
        ("user", "Hi there!"),
        ("assistant", "Hello! How can I help you today?"),
        ("user", "My name is Priya Raman and I work at Nimbus Analytics."),
        ("assistant", "Nice to meet you, Priya. What do you do at Nimbus Analytics?"),
        ("user", "I build dashboards mostly. The weather in town has been grey all week."),
        ("assistant", "Grey weeks are tough. Anything I can help with on the dashboards?"),
        ("user", "I live in Berlin, so grey is normal here."),
        ("user", "Can you explain what a left join does in SQL?"),
        ("assistant", "A left join keeps every row from the left table and matches rows from "
                      "the right table where the join condition holds, filling gaps with NULL."),
        ("user", "Thanks. Also I'm allergic to peanuts, which matters for the team lunch."),
        ("user", "I prefer tea over coffee."),
        ("assistant", "Noted. Tea it is."),
        ("user", "ok bye"),
    ]),
    (3, [
        ("user", "Good morning."),
        ("user", "My dog is called Biscuit and he chewed my charger today."),
        ("assistant", "Oh no, poor charger. Is Biscuit okay?"),
        ("user", "He is fine. We are migrating our data pipeline to Kafka this quarter."),
        ("assistant", "Kafka is a solid choice for event streaming. What volume do you expect?"),
        ("user", "About 40000 events per second at peak, so partitioning matters a lot."),
        ("user", "Remember that my passport expires in March 2027."),
        ("assistant", "I will remember that your passport expires in March 2027."),
        ("user", "My favorite book is Dune, I reread it every winter."),
        ("user", "cool thanks"),
    ]),
    (10, [
        ("user", "Big news: I moved to Lisbon last week!"),
        ("assistant", "Congratulations! How are you finding Lisbon?"),
        ("user", "Sunny and hilly. I love hiking, and the coast trails are amazing."),
        ("user", "I am vegetarian, so the food scene took some exploring."),
        ("assistant", "Lisbon has great vegetarian options these days."),
        ("user", "The Kafka migration slipped because of a schema registry issue."),
        ("assistant", "Schema registry issues are common; versioning schemas helps."),
        ("user", "thanks"),
    ]),
    (20, [
        ("user", "I no longer work at Nimbus Analytics."),
        ("user", "I just joined Helix Robotics as a staff engineer."),
        ("assistant", "Exciting! What will you work on at Helix Robotics?"),
        ("user", "Perception pipelines for warehouse robots. I'm a bit worried about the "
                 "learning curve."),
        ("assistant", "That is normal; your data pipeline experience will transfer."),
        ("user", "My birthday is on June 5, the team wants to celebrate."),
        ("user", "okay"),
    ]),
]

QUERY_DAY = 35.0


@dataclass(frozen=True)
class Probe:
    query: str
    expected: tuple[str, ...]          # any of these substrings marks a relevant memory
    stale: tuple[str, ...] = ()        # outdated values that must not outrank the current one


PROBES: list[Probe] = [
    Probe("What is the user's name?", ("Priya",)),
    Probe("Which city does the user live in?", ("Lisbon",), ("Berlin",)),
    Probe("Which company is the user's employer?", ("Helix",), ("Nimbus",)),
    Probe("Does the user have any food allergies?", ("peanut",)),
    Probe("Does the user prefer tea or coffee?", ("tea over coffee", "Tea it is")),
    Probe("What is the name of the user's dog?", ("Biscuit",)),
    Probe("When does the user's passport expire?", ("March 2027",)),
    Probe("What is the user's favorite book?", ("Dune",)),
    Probe("What diet does the user follow?", ("vegetarian",)),
    Probe("When is the user's birthday?", ("June 5",)),
    Probe("What outdoor hobby does the user enjoy?", ("hiking",)),
    Probe("What data pipeline migration did we discuss?", ("Kafka",)),
]
