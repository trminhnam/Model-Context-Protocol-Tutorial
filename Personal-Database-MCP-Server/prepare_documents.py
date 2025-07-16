from datasets import load_dataset
import os
import json
from datasets.utils.logging import set_verbosity_error
import tqdm

set_verbosity_error()

CACHE_DIR = "./cache"
DOCUMENT_DIR = "./documents"

dataset_ids = [
    "burgerbee/pedagogy_textbook",
    "burgerbee/pedagogy_wiki",
    "burgerbee/psychology_textbook",
    "burgerbee/psychology_wiki",
    "burgerbee/psychiatry_textbook",
    "burgerbee/psychiatry_wiki",
    "burgerbee/art_and_culture_textbook",
    "burgerbee/art_and_culture_wiki",
    "burgerbee/medicine_textbook",
    "burgerbee/medicine_wiki",
    "burgerbee/chemistry_textbook",
    "burgerbee/chemistry_wiki",
    "burgerbee/social_studies_textbook",
    "burgerbee/social_studies_wiki",
    "burgerbee/religion_textbook",
    "burgerbee/religion_wiki",
    "burgerbee/science_studies_textbook",
    "burgerbee/science_studies_wiki",
    "burgerbee/history_textbook",
    "burgerbee/history_wiki",
    "burgerbee/philosophy_textbook",
    "burgerbee/philosophy_wiki",
    "burgerbee/biology_textbook",
    "burgerbee/biology_wiki",
    "burgerbee/physics_textbook",
    "burgerbee/physics_wiki",
]


for dataset_id in tqdm.tqdm(dataset_ids, desc="Downloading datasets"):
    print(f"Downloading {dataset_id}...")
    dataset = load_dataset(dataset_id, cache_dir=CACHE_DIR)
    print(f"Downloaded {dataset_id} with {len(dataset)} splits.")

    dataset_dir = os.path.join(DOCUMENT_DIR, dataset_id.split("/")[1])
    os.makedirs(dataset_dir, exist_ok=True)

    for split in dataset:
        print(f" - {split}: {len(dataset[split])} examples")
        split_data = len(dataset[split])

        for i, sample in enumerate(dataset[split]):
            title, text = sample["title"], sample["text"]
            source = f"https://hf.co/datasets/{dataset_id}"
            sample["source"] = source
            sample["split"] = split
            sample["id"] = f"{dataset_id}-{split}-{i}"

            doc_path = os.path.join(dataset_dir, f"{title.replace('/', '_')}_{i}.json")
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=4)

        print(f"Saved {split_data} samples to {dataset_dir}")
    print(f"Finished processing {dataset_id}.\n")
print("All datasets processed successfully.")
