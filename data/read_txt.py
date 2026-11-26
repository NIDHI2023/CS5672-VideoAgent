from itertools import islice
from pathlib import Path
import json
file_path = Path("AGQA_balanced") / "test_balanced.txt"
print(file_path.resolve())

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total questions:", len(data))

for question_id, item in list(data.items())[:5]:
    print("ID:", question_id)
    print("Video:", item["video_id"])
    print("Question:", item["question"])
    print("Answer:", item["answer"])
    print("Type:", item["ans_type"])
    print()



from collections import Counter

print("Answer types:")
print(Counter(q["ans_type"] for q in data.values()))

print("\nSemantic categories:")
print(Counter(q["semantic"] for q in data.values()))

print("\nStructural categories:")
print(Counter(q["structural"] for q in data.values()))

print("\nUnique videos:")
print(len(set(q["video_id"] for q in data.values())))