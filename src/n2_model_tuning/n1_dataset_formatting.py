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

def build_dataset(images_dir: str):
    dataset = []
    root = Path(images_dir)
    for image_path in root.rglob("*.jpg"):
        parts = image_path.parts # Devuelve las partes del path como un array.
        species = parts[-3]
        description = parts[-2]

        image = Image.open(image_path).convert("RGB")
        answer = (
            f"**Species:** { _format_species(species) }\n"
            f"Showing **{ _format_description(description) }**"
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    { "type": "image", "image": image },
                    { "type": "text", "text": "Identify this plant's species name and, if possible, what part of it is being shown." }
                ]
            },
            {
                "role": "assistant",
                "content": [{ "type": "text", "text": answer }]
            }
        ]

        dataset.append({
            "messages": conversation
        })
    return dataset
    

if __name__ == "__main__":
    build_dataset("data/raw/images")