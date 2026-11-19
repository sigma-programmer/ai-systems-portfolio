"""Bundled sample corpus: a fictional industry of companies, people and places.

Facts are deliberately scattered across documents so that multi-hop questions
("Who is the CEO of the company that acquired Quillon Systems?") need 2-3
documents, and documents that *look* most similar to the question often name
a distractor (Quillon's own former CEO, for instance). The corpus also
exercises entity resolution: a bare surname ("Okafor"), a bare company stem
("Zentrix"), an acronym alias ("Nimbus Freight (NF)"), and a typo
("Quilon Systems").

All names are invented.
"""
from __future__ import annotations

from pydantic import BaseModel

from .models import Document

_RAW: list[tuple[str, str, str]] = [
    ("d01", "Zentrix Labs is founded",
     "Zentrix Labs was founded in 2014 by Alice Moreau and Ben Okafor. The company designs low-power AI "
     "accelerators for edge devices. Zentrix Labs is headquartered in Port Alder. Alice Moreau serves as CEO of "
     "Zentrix Labs, while Ben Okafor is the CTO of Zentrix Labs."),
    ("d02", "The Atlas Chip",
     "In 2019, Zentrix Labs launched the Atlas Chip, a neural processor for drones and cameras. Analysts praised "
     "the chip for its energy efficiency. Ferrox Materials supplies Zentrix Labs with the rare-earth substrates "
     "used in the processor."),
    ("d03", "Quillon Systems profile",
     "Quillon Systems was founded by Dmitri Vasquez in 2011. Quillon Systems is based in Northvale. The company "
     "developed the Lumen Sensor, an optical sensor used in industrial inspection. Dmitri Vasquez was the CEO of "
     "Quillon Systems for a decade."),
    ("d04", "Zentrix Labs acquires Quillon Systems",
     "Zentrix Labs acquired Quillon Systems in 2021 for an undisclosed sum. The deal gave the buyer access to the "
     "Lumen Sensor product line. Dmitri Vasquez, the founder of Quillon Systems, left after the acquisition "
     "closed."),
    ("d05", "Halvorsen Group at a glance",
     "Halvorsen Group is a diversified industrial conglomerate headquartered in Eastmere. Erik Halvorsen founded "
     "Halvorsen Group in 1978. Ingrid Halvorsen is the chief executive of Halvorsen Group. Elena Petrova serves "
     "as CFO of Halvorsen Group."),
    ("d06", "Halvorsen Group buys Zentrix Labs",
     "Halvorsen Group acquired Zentrix Labs in 2023, its largest technology deal to date. Alice Moreau remained "
     "chief executive of Zentrix after the acquisition. Brightwater Capital, an early investor, sold its stake as "
     "part of the deal."),
    ("d07", "Brightwater Capital",
     "Brightwater Capital is a venture firm based in Port Alder. Marcus Lee founded Brightwater Capital in 2009. "
     "Marcus Lee is the managing partner of Brightwater Capital. Brightwater Capital invested in Zentrix Labs "
     "during its seed round."),
    ("d08", "Brightwater backs Verdant Bio",
     "Brightwater Capital invested in Verdant Bio in 2020. Verdant Bio was founded by Hana Sato, who also serves "
     "as its CEO. Verdant Bio is headquartered in Kestrel Bay and engineers drought-resistant crops."),
    ("d09", "Verdant Bio's founder",
     "Before founding the startup, Hana Sato worked as a plant geneticist at Arden University. Verdant Bio opened "
     "a second greenhouse campus outside Kestrel Bay in 2022."),
    ("d10", "Orinoco Robotics profile",
     "Orinoco Robotics was founded by Rafael Costa in 2015. The firm builds collaborative robot arms for "
     "warehouses. Orinoco Robotics is headquartered in Northvale. Priya Raman is the CEO of Orinoco Robotics and "
     "Victor Huang is its CTO."),
    ("d11", "The Kite Arm",
     "Orinoco Robotics released the Kite Arm in 2020. The Kite Arm runs on processors designed in Port Alder. "
     "Orinoco Robotics partnered with Zentrix Labs to co-develop motion control software."),
    ("d12", "Orinoco Robotics acquires Tessel Analytics",
     "Orinoco Robotics acquired Tessel Analytics in 2022 to strengthen its fleet analytics. Tessel Analytics had "
     "been founded by Sofia Lindqvist. Tessel Analytics is based in Kestrel Bay."),
    ("d13", "Sofia Lindqvist moves to energy",
     "After the acquisition, Sofia Lindqvist joined Caldera Energy as chief scientist. She leads the grid "
     "forecasting team at Caldera Energy."),
    ("d14", "Caldera Energy profile",
     "Caldera Energy was founded by Grace Mbeki in 2008. Olu Adeyemi became CEO of Caldera Energy in 2018. "
     "Caldera Energy is headquartered in Redmoor and operates wind farms along the coast."),
    ("d15", "Caldera Energy acquires Solace Grid",
     "Caldera Energy acquired Solace Grid, a battery storage startup, in 2021. Solace Grid was founded by Ken "
     "Watanabe. Solace Grid is based in Redmoor."),
    ("d16", "Nimbus Freight",
     "Nimbus Freight (NF) is a subsidiary of Halvorsen Group. Tomas Herrera is the CEO of Nimbus Freight. NF is "
     "headquartered in Eastmere and runs a fleet of autonomous trucks."),
    ("d17", "Nimbus Freight partners with Orinoco Robotics",
     "Nimbus Freight partnered with Orinoco Robotics in 2023 to automate loading docks. The pilot deployed Kite "
     "Arm robots at three depots near Eastmere."),
    ("d18", "Ferrox Materials",
     "Ferrox Materials was founded by Leila Haddad. Leila Haddad is the CEO of Ferrox Materials. Ferrox "
     "Materials is located in Lowford, a short drive from Arden University."),
    ("d19", "Arden University alumni",
     "Arden University is located in Lowford. Alice Moreau studied at Arden University before starting her "
     "company. Priya Raman graduated from Arden University with a degree in mechanical engineering."),
    ("d20", "Edge AI outlook",
     "Analysts expect edge AI spending to double by 2027. Zentrix Labs, Orinoco Robotics and Quilon Systems are "
     "frequently cited as leaders in the Northvale and Port Alder technology corridor."),
    ("d21", "Storage boom in Redmoor",
     "Grid-scale storage is booming in Redmoor. Caldera Energy and Solace Grid together control most of the "
     "battery capacity in the region. Olu Adeyemi said Caldera Energy plans to double storage by 2026."),
    ("d22", "Kestrel Bay agritech hub",
     "Kestrel Bay has become a hub for agricultural technology. Verdant Bio and Tessel Analytics both grew out "
     "of research programs in the city. Hana Sato said Verdant Bio will hire 200 engineers."),
    ("d23", "Quillon Systems after the deal",
     "Following the purchase, Quillon Systems became a division of Zentrix Labs. Okafor now oversees the Lumen "
     "Sensor roadmap."),
    ("d24", "Halvorsen Group strategy",
     "Ingrid Halvorsen said Halvorsen Group will expand into logistics and semiconductors. Under Ingrid "
     "Halvorsen, Halvorsen Group has completed six acquisitions."),
    ("d25", "Ferrox Materials supply deal",
     "Ferrox Materials signed a five-year supply agreement with Zentrix Labs in 2022. The agreement makes Ferrox "
     "Materials the main substrate supplier for the Atlas Chip."),
    ("d26", "Quillon Systems leadership change",
     "When Dmitri Vasquez stepped down, the board of Quillon Systems praised his decade as CEO. Quillon Systems "
     "employees in Northvale were offered retention bonuses."),
]

DOCUMENTS: list[Document] = [Document(id=i, title=t, text=x) for i, t, x in _RAW]

#: Ground-truth typed triples stated in the corpus (canonical names), for
#: measuring extraction precision / recall.
GOLD_TRIPLES: set[tuple[str, str, str]] = {
    ("Alice Moreau", "FOUNDED", "Zentrix Labs"),
    ("Ben Okafor", "FOUNDED", "Zentrix Labs"),
    ("Zentrix Labs", "LOCATED_IN", "Port Alder"),
    ("Alice Moreau", "CEO_OF", "Zentrix Labs"),
    ("Ben Okafor", "CTO_OF", "Zentrix Labs"),
    ("Zentrix Labs", "PRODUCES", "Atlas Chip"),
    ("Ferrox Materials", "SUPPLIES", "Zentrix Labs"),
    ("Dmitri Vasquez", "FOUNDED", "Quillon Systems"),
    ("Quillon Systems", "LOCATED_IN", "Northvale"),
    ("Quillon Systems", "PRODUCES", "Lumen Sensor"),
    ("Dmitri Vasquez", "CEO_OF", "Quillon Systems"),
    ("Zentrix Labs", "ACQUIRED", "Quillon Systems"),
    ("Halvorsen Group", "LOCATED_IN", "Eastmere"),
    ("Erik Halvorsen", "FOUNDED", "Halvorsen Group"),
    ("Ingrid Halvorsen", "CEO_OF", "Halvorsen Group"),
    ("Elena Petrova", "CFO_OF", "Halvorsen Group"),
    ("Halvorsen Group", "ACQUIRED", "Zentrix Labs"),
    ("Brightwater Capital", "LOCATED_IN", "Port Alder"),
    ("Marcus Lee", "FOUNDED", "Brightwater Capital"),
    ("Marcus Lee", "WORKS_AT", "Brightwater Capital"),
    ("Brightwater Capital", "INVESTED_IN", "Zentrix Labs"),
    ("Brightwater Capital", "INVESTED_IN", "Verdant Bio"),
    ("Hana Sato", "FOUNDED", "Verdant Bio"),
    ("Hana Sato", "CEO_OF", "Verdant Bio"),
    ("Verdant Bio", "LOCATED_IN", "Kestrel Bay"),
    ("Hana Sato", "WORKS_AT", "Arden University"),
    ("Rafael Costa", "FOUNDED", "Orinoco Robotics"),
    ("Orinoco Robotics", "LOCATED_IN", "Northvale"),
    ("Priya Raman", "CEO_OF", "Orinoco Robotics"),
    ("Victor Huang", "CTO_OF", "Orinoco Robotics"),
    ("Orinoco Robotics", "PRODUCES", "Kite Arm"),
    ("Orinoco Robotics", "PARTNERED_WITH", "Zentrix Labs"),
    ("Orinoco Robotics", "ACQUIRED", "Tessel Analytics"),
    ("Sofia Lindqvist", "FOUNDED", "Tessel Analytics"),
    ("Tessel Analytics", "LOCATED_IN", "Kestrel Bay"),
    ("Sofia Lindqvist", "WORKS_AT", "Caldera Energy"),
    ("Grace Mbeki", "FOUNDED", "Caldera Energy"),
    ("Olu Adeyemi", "CEO_OF", "Caldera Energy"),
    ("Caldera Energy", "LOCATED_IN", "Redmoor"),
    ("Caldera Energy", "ACQUIRED", "Solace Grid"),
    ("Ken Watanabe", "FOUNDED", "Solace Grid"),
    ("Solace Grid", "LOCATED_IN", "Redmoor"),
    ("Nimbus Freight", "SUBSIDIARY_OF", "Halvorsen Group"),
    ("Tomas Herrera", "CEO_OF", "Nimbus Freight"),
    ("Nimbus Freight", "LOCATED_IN", "Eastmere"),
    ("Nimbus Freight", "PARTNERED_WITH", "Orinoco Robotics"),
    ("Leila Haddad", "FOUNDED", "Ferrox Materials"),
    ("Leila Haddad", "CEO_OF", "Ferrox Materials"),
    ("Ferrox Materials", "LOCATED_IN", "Lowford"),
    ("Arden University", "LOCATED_IN", "Lowford"),
    ("Alice Moreau", "ALUMNUS_OF", "Arden University"),
    ("Priya Raman", "ALUMNUS_OF", "Arden University"),
    ("Quillon Systems", "SUBSIDIARY_OF", "Zentrix Labs"),
    ("Ferrox Materials", "SUPPLIES", "Atlas Chip"),
}


class QAItem(BaseModel):
    """A benchmark question with its gold answer and supporting documents."""

    question: str
    answer: str
    hops: int
    support: list[str]


QUESTIONS: list[QAItem] = [
    # single-hop controls
    QAItem(question="Who founded Quillon Systems?", answer="Dmitri Vasquez", hops=1, support=["d03"]),
    QAItem(question="Where is Caldera Energy headquartered?", answer="Redmoor", hops=1, support=["d14"]),
    QAItem(question="Who is the CEO of Orinoco Robotics?", answer="Priya Raman", hops=1, support=["d10"]),
    QAItem(question="Which company acquired Tessel Analytics?", answer="Orinoco Robotics", hops=1, support=["d12"]),
    # two-hop
    QAItem(question="Who is the CEO of the company that acquired Quillon Systems?", answer="Alice Moreau", hops=2, support=["d04", "d01"]),
    QAItem(question="Who is the CEO of the company that acquired Zentrix Labs?", answer="Ingrid Halvorsen", hops=2, support=["d06", "d05"]),
    QAItem(question="Where is the company that acquired Tessel Analytics headquartered?", answer="Northvale", hops=2, support=["d12", "d10"]),
    QAItem(question="Who founded the company that acquired Solace Grid?", answer="Grace Mbeki", hops=2, support=["d15", "d14"]),
    QAItem(question="Where is the company founded by Dmitri Vasquez headquartered?", answer="Northvale", hops=2, support=["d03"]),
    QAItem(question="Who founded the company whose CEO is Priya Raman?", answer="Rafael Costa", hops=2, support=["d10"]),
    QAItem(question="Which city is the parent company of Nimbus Freight headquartered in?", answer="Eastmere", hops=2, support=["d16", "d05"]),
    QAItem(question="Who is the CEO of the company that Ferrox Materials supplies?", answer="Alice Moreau", hops=2, support=["d02", "d01"]),
    QAItem(question="What product did the company acquired by Zentrix Labs develop?", answer="Lumen Sensor", hops=2, support=["d04", "d03"]),
    QAItem(question="Who founded the company that invested in Verdant Bio?", answer="Marcus Lee", hops=2, support=["d08", "d07"]),
    QAItem(question="Who is the CEO of the company that partnered with Nimbus Freight?", answer="Priya Raman", hops=2, support=["d17", "d10"]),
    QAItem(question="Who is the CTO of the company that acquired Quillon Systems?", answer="Ben Okafor", hops=2, support=["d04", "d01"]),
    QAItem(question="Who is the CFO of the company that acquired Zentrix Labs?", answer="Elena Petrova", hops=2, support=["d06", "d05"]),
    QAItem(question="What product did the company whose CTO is Victor Huang develop?", answer="Kite Arm", hops=2, support=["d10", "d11"]),
    QAItem(question="Who is the CEO of the parent company of Nimbus Freight?", answer="Ingrid Halvorsen", hops=2, support=["d16", "d05"]),
    QAItem(question="Where is the company that developed the Atlas Chip headquartered?", answer="Port Alder", hops=2, support=["d02", "d01"]),
    # three-hop
    QAItem(question="Who is the CEO of the company that acquired the company that acquired Quillon Systems?", answer="Ingrid Halvorsen", hops=3, support=["d04", "d06", "d05"]),
    QAItem(question="Where is the company that acquired the company founded by Ken Watanabe headquartered?", answer="Redmoor", hops=3, support=["d15", "d14"]),
    QAItem(question="Where did the CEO of the company that partnered with Nimbus Freight study?", answer="Arden University", hops=3, support=["d17", "d10", "d19"]),
    QAItem(question="Where is the company that the founder of Tessel Analytics joined headquartered?", answer="Redmoor", hops=3, support=["d12", "d13", "d14"]),
    QAItem(question="Who is the CFO of the company that acquired the company that acquired Quillon Systems?", answer="Elena Petrova", hops=3, support=["d04", "d06", "d05"]),
    QAItem(question="Who founded the company that invested in the company that acquired Quillon Systems?", answer="Marcus Lee", hops=3, support=["d04", "d07"]),
]

GLOBAL_QUESTIONS: list[str] = [
    "Which companies operate in energy and battery storage?",
    "Summarize the major acquisitions in this industry.",
    "What are the main themes across the companies based in Kestrel Bay?",
]
