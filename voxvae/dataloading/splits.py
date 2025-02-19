from typing import List


def split_counts(total: int, percentages: List[int]) -> List[int]:
    # Ensure percentages sum to 100
    if sum(percentages) == 1:
        percentages = [p*100 for p in percentages]
    assert sum(percentages) == 100

    # Initial allocation based on percentage
    raw_counts = [max(1, round(total * p / 100)) for p in percentages]

    # Adjust to match total exactly
    difference = total - sum(raw_counts)

    # Distribute the remaining items to the largest percentage groups
    for _ in range(abs(difference)):
        if difference > 0:
            # Add 1 to the split with the highest percentage
            idx = max(range(len(percentages)), key=lambda i: (percentages[i], -raw_counts[i]))
            raw_counts[idx] += 1
        elif difference < 0:
            # Remove 1 from the split with the highest allocated count (while ensuring min 1)
            idx = max((i for i in range(len(raw_counts)) if raw_counts[i] > 1),
                      key=lambda i: raw_counts[i])
            raw_counts[idx] -= 1

    return raw_counts
