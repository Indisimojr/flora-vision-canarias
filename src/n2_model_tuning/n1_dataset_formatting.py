from pathlib import Path
from PIL import Image

def _format_species(species: str):
    words = species.split("_")
    
    words[0] = words[0].capitalize()
    return " ".join(words)

def _format_description(description: str):
    if description == "habit":
        return "full plant"
    else: return description

def build_dataset(corpus_dir: str, images_dir: str):
    dataset = []
    
    corpus_plants: dict = {}
    corpus_root = Path(corpus_dir)
    for corpus_path in corpus_root.rglob("*.md"): # Recogemos los archivos terminados en ".md".
        species = corpus_path.parts[-2] # Recogemos la parte del path que guarda la especie.

        if not corpus_plants.get(species):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_plants[species] = "\n".join(f.readlines())
    
    image_root = Path(images_dir)
    for image_path in image_root.rglob("*.jpg"):
        parts = image_path.parts # Devuelve las partes del path como un array.
        species = parts[-3]
        description = parts[-2]

        image = Image.open(image_path).convert("RGB")
        answer = (
            f"**Species:** { _format_species(species) }, an endemic plant of the Canary Islands.\n"
            f"Showing **{ _format_description(description) }**.\n"
            f"{ corpus_plants[species] }"
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    { "type": "image", "image": image },
                    { "type": "text", "text": "Identify the species of this endemic planta from the Canary Islands and provide details about it." }
                ]
            },
            {
                "role": "assistant",
                "content": [{ "type": "text", "text": answer }]
            }
        ]

        dataset.append(
            {"messages": conversation}
        )
    return dataset

if __name__ == "__main__":
    import re
    from collections import Counter

    dataset = build_dataset("data/raw/corpus", "data/raw/images")
    
    def extract_label(sample):
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                
                match = re.search(r"\*\*Species:\*\*\s+(.+?),", msg["content"][0]["text"])

                if match:
                    return match.group(1).strip()
    
    labels = [extract_label(sample) for sample in dataset]
    
    label_counts = Counter(labels)
    print(f"Classes found: {len(label_counts)}")
    print(f"Samples per class: {label_counts.most_common()}")