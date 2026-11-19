"""Bundled knowledge base: a fictional company handbook plus general-knowledge notes.

"Lumora Labs" is invented for this demo. Every fact used by the benchmark is
stated in exactly one document so citation precision can be measured.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


_RAW: list[tuple[str, str, str]] = [
    ("lumora-overview", "Lumora Labs overview",
     "Lumora Labs is a consumer electronics company founded in 2014 by Dana Whitfield. "
     "The company is headquartered in Oslo, Norway. Lumora Labs employs roughly 420 people "
     "across three offices. Its mission is to build durable, repairable smart-home devices."),
    ("lumora-ceo", "Leadership",
     "Dana Whitfield has served as chief executive officer of Lumora Labs since 2014. "
     "Before Lumora, Whitfield led hardware engineering at a satellite startup for six years. "
     "In 2022 Whitfield was named Nordic Founder of the Year."),
    ("beacon-specs", "Beacon lamp specifications",
     "The Beacon is the flagship smart lamp made by Lumora. The Beacon battery lasts 18 hours on a "
     "full charge and recharges in 2 hours over USB-C. The Beacon weighs 640 grams. "
     "The Beacon is not waterproof and should not be used outdoors in rain."),
    ("beacon-pricing", "Beacon pricing",
     "The Beacon retails for 149 dollars in the United States. A bundle of two Beacon lamps "
     "costs 269 dollars. Educational institutions receive a 15 percent discount on bulk orders."),
    ("warranty", "Warranty policy",
     "All Lumora devices include a 2 year limited warranty. The warranty covers manufacturing "
     "defects but does not cover water damage or accidental drops. Warranty claims must be filed "
     "through the Lumora support portal."),
    ("refunds", "Refund policy",
     "Customers may return any Lumora product within 30 days of delivery for a full refund. "
     "Refunds are issued to the original payment method within 5 business days. Opened software "
     "subscriptions are not eligible for refunds."),
    ("pto", "Paid time off",
     "Full-time Lumora employees receive 25 days of paid time off per year. Unused paid time off "
     "can carry over up to a maximum of 5 days into the next year. Employees also receive 10 "
     "public holidays."),
    ("remote-work", "Remote work policy",
     "Lumora allows employees to work remotely up to 3 days per week. Engineering teams gather in "
     "the Oslo office every Tuesday. Remote work from outside Norway requires approval from the "
     "people team."),
    ("parental-leave", "Parental leave",
     "Lumora offers 16 weeks of fully paid parental leave to all new parents. Parental leave can "
     "be taken within the first 18 months after a birth or adoption."),
    ("support", "Customer support",
     "Lumora customer support is available from 8 am to 8 pm Central European Time on weekdays. "
     "Support is offered by chat, email, and phone. Average first response time on chat is under "
     "2 minutes."),
    ("security", "Security practices",
     "Lumora devices encrypt all data in transit using TLS 1.3. Firmware updates are "
     "cryptographically signed. Lumora runs a public bug bounty program that pays up to 20000 "
     "dollars for critical vulnerabilities."),
    ("pulse-specs", "Pulse thermostat",
     "The Pulse is the smart thermostat made by Lumora, released in 2019. The Pulse supports "
     "Zigbee and Thread but does not support Z-Wave. Lumora reports that the Pulse reduces "
     "household heating costs by an average of 12 percent."),
    ("sustainability", "Sustainability",
     "Lumora committed to carbon neutral operations by 2030. In 2023 the company sourced 80 "
     "percent of its office electricity from renewable energy. Packaging for all devices is made "
     "from recycled cardboard."),
    ("funding", "Funding and financials",
     "Lumora Labs raised a 40 million dollar Series B round in 2021 led by Fjord Ventures. The "
     "company reported revenue of 95 million dollars in 2023. Lumora has been profitable since "
     "2022."),
    ("photosynthesis", "Photosynthesis",
     "Photosynthesis is the process by which plants convert light energy into chemical energy. "
     "It takes place mainly in the chloroplasts of leaf cells. Photosynthesis produces oxygen as "
     "a by-product and consumes carbon dioxide."),
    ("mars", "Mars",
     "Mars is the fourth planet from the Sun. A day on Mars lasts about 24.6 hours. Mars has two "
     "small moons named Phobos and Deimos. The surface gravity of Mars is about 38 percent of "
     "the gravity on Earth."),
    ("moon", "The Moon",
     "The Moon is the only natural satellite of Earth. It orbits Earth at an average distance of "
     "about 384400 kilometers. The Moon has no atmosphere capable of supporting liquid water."),
    ("everest", "Mount Everest",
     "Mount Everest is the highest mountain above sea level, at 8849 meters. It sits on the "
     "border between Nepal and China. The first confirmed ascent was made in 1953 by Edmund "
     "Hillary and Tenzing Norgay."),
    ("python", "Python language",
     "Python is a high-level programming language first released in 1991. It was created by "
     "Guido van Rossum. Python emphasizes code readability and uses indentation to delimit "
     "blocks."),
    ("coffee", "Coffee",
     "Coffee beans are the roasted seeds of the Coffea plant. The two most commercially "
     "important species are arabica and robusta. Robusta beans contain roughly twice as much "
     "caffeine as arabica beans."),
    ("honeybees", "Honeybees",
     "Honeybees communicate the location of food sources through a waggle dance. A single "
     "honeybee colony can contain up to 60000 workers. Worker bees live for about six weeks "
     "during the summer."),
    ("great-wall", "Great Wall of China",
     "The Great Wall of China stretches more than 21000 kilometers including all its branches. "
     "Most of the existing wall was built during the Ming dynasty. The Great Wall is not visible "
     "to the naked eye from low Earth orbit."),
]

CORPUS: list[Document] = [Document(d, t, x) for d, t, x in _RAW]


def load_corpus() -> list[Document]:
    """Return a fresh copy of the bundled corpus."""
    return list(CORPUS)
