import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/original_annotation/dailydialog_valid.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Initialize counters
emotion_counter = Counter()
speaker_counter = Counter()
dialogue_lengths = []

# Process dataset
for dialogue_id, conversations in data.items():
    for dialogue in conversations:
        dialogue_lengths.append(len(dialogue))
        for turn in dialogue:
            emotion_counter[turn["emotion"]] += 1
            speaker_counter[turn["speaker"]] += 1

# Summary statistics
print("Total dialogues:", len(data))
print("Average dialogue length:", sum(dialogue_lengths) / len(dialogue_lengths))
print("Emotion distribution:", emotion_counter)
print("Speaker distribution:", speaker_counter)

# Visualization
plt.figure(figsize=(12, 5))
sns.barplot(x=list(emotion_counter.keys()), y=list(emotion_counter.values()), palette="viridis")
plt.xticks(rotation=45)
plt.title("Emotion Distribution in Dataset Valid")
plt.xlabel("Emotions")
plt.ylabel("Frequency")
plt.show()
