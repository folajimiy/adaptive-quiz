# core/skill_graph.py

from typing import Dict, List

# These are your actual topics from the dataset
ALL_TOPICS: List[str] = [
    "Java Fundamentals",
    "Control Flow",
    "Loops",
    "Arrays",
    "Strings",
    "Methods",
    "Objects and Classes",
    "Encapsulation",
    "Inheritance",
    "Polymorphism",
    "Abstract Classes",
    "Interfaces",
    "Generics",
    "Collections",
    "JavaFX",
    "Event-Driven Programming",
]

# Topic-level prerequisites (Skill Graph)
TOPIC_PREREQS: Dict[str, List[str]] = {
    "Java Fundamentals": [],
    "Objects and Classes": ["Java Fundamentals"],
    "Methods": ["Java Fundamentals", "Objects and Classes"],
    "Control Flow": ["Java Fundamentals"],
    "Loops": ["Control Flow"],
    "Arrays": ["Loops", "Control Flow"],
    "Strings": ["Java Fundamentals"],
    "Encapsulation": ["Objects and Classes"],
    "Inheritance": ["Objects and Classes"],
    "Polymorphism": ["Inheritance"],
    "Abstract Classes": ["Inheritance"],
    "Interfaces": ["Abstract Classes", "Polymorphism"],
    "Generics": ["Collections"],
    "Collections": ["Arrays", "Objects and Classes"],
    "Event-Driven Programming": ["Java Fundamentals", "Objects and Classes"],
    "JavaFX": ["Event-Driven Programming"],
}


def get_prereq_topics(topic: str) -> List[str]:
    """Return the list of prerequisite topics for a given topic."""
    return TOPIC_PREREQS.get(topic, [])


def get_dependent_topics(topic: str) -> List[str]:
    """Return topics that depend on the given topic."""
    dependents = []
    for t, prereqs in TOPIC_PREREQS.items():
        if topic in prereqs:
            dependents.append(t)
    return dependents
