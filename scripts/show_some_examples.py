# Script to load dataset in streaming mode and show some examples
from datasets import load_dataset


def pretty_print_query(item: dict) -> None:
    """Pretty print a query item with id, query, answers, and tools."""
    import json

    print(f"\n{'='*80}")
    print(f"ID: {item['id']}")
    print(f"\nQuery: {item['query']}")

    print(f"\n{'─'*80}")
    print("Answers:")
    answers = json.loads(item['answers'])
    for i, answer in enumerate(answers, 1):
        print(f"  {i}. {answer['name']}")
        print(f"     Arguments: {json.dumps(answer['arguments'], indent=8)}")

    print(f"\n{'─'*80}")
    print("Available Tools:")
    tools = json.loads(item['tools'])
    for tool in tools:
        print(f"\n  • {tool['name']}")
        print(f"    Description: {tool['description']}")
        print(f"    Parameters:")
        for param_name, param_info in tool['parameters'].items():
            print(f"      - {param_name}: {param_info['type']}")
            print(f"        {param_info['description']}")
            if 'default' in param_info:
                print(f"        Default: {param_info['default']}")
    print(f"{'='*80}\n")


def show_some_examples(n: int=5):
    import random

    ds = load_dataset("Salesforce/xlam-function-calling-60k", streaming=True, split="train")

    # Take a larger sample to randomize from
    sample_size = max(n * 10, 100)
    sample = list(ds.take(sample_size))

    # Randomly select n examples
    selected = random.sample(sample, min(n, len(sample)))

    for item in selected:
        pretty_print_query(item)


if __name__ == "__main__":
    show_some_examples()
