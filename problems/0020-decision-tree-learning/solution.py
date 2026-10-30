import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""

    counts = Counter(labels)
    total = len(labels)

    entropy = 0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""

    labels = [example[target_attr] for example in examples]

    total_entropy = calculate_entropy(labels)

    total = len(examples)
    weighted_entropy = 0

    values = sorted(set(example[attr] for example in examples))

    for value in values:

        subset = [
            example for example in examples
            if example[attr] == value
        ]

        subset_labels = [
            example[target_attr] for example in subset
        ]

        weighted_entropy += (
            len(subset) / total
        ) * calculate_entropy(subset_labels)

    return total_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""

    counts = Counter(
        example[target_attr] for example in examples
    )

    return sorted(
        counts,
        key=lambda x: (-counts[x], x)
    )[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""

    # Base case: empty examples
    if not examples:
        return None

    # Base case: all examples have the same class
    classes = set(
        example[target_attr] for example in examples
    )

    if len(classes) == 1:
        return next(iter(classes))

    # Base case: no attributes remaining
    if not attributes:
        return majority_class(examples, target_attr)

    # Select attribute with maximum information gain
    best_attr = attributes[0]
    best_gain = calculate_information_gain(
        examples, best_attr, target_attr
    )

    for attr in attributes[1:]:

        gain = calculate_information_gain(
            examples, attr, target_attr
        )

        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    # Create tree
    tree = {best_attr: {}}

    remaining_attributes = [
        attr for attr in attributes
        if attr != best_attr
    ]

    # Split examples based on selected attribute
    values = sorted(
        set(example[best_attr] for example in examples)
    )

    for value in values:

        subset = [
            example for example in examples
            if example[best_attr] == value
        ]

        tree[best_attr][value] = learn_decision_tree(
            subset,
            remaining_attributes,
            target_attr
        )

    return tree