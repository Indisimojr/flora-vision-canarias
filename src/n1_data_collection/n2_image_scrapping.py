### Importación de librerías y datos ###
import pandas as pd
from scrapling.fetchers import AsyncStealthySession
from scrapling.spiders import Spider, Response, Request
import aiofiles
import re
import os

df = pd.read_json("data/raw/flower_media.json")

### Elección de flores ###
endemic_plants = [
    "Echium decaisnei",
    "Lobularia canariensis",
    "Salvia canariensis",
    "Carlina salicifolia",
    "Aichryson laxum",
    "Juniperus cedrus",
    "Rumex lunaria",
    "Ranunculus cortusifolius",
    "Lavandula canariensis",
    "Pinus canariensis",
    "Ilex canariensis",
    "Arbutus canariensis",
    "Echium wildpretii",
    "Hypericum canariense",
    "Euphorbia aphylla",
    "Aizoon canariense",
    "Pterocephalus lasiospermus",
    "Asteriscus sericeus",
    "Aeonium arboreum",
    "Artemisia thuscula",
    "Euphorbia lamarckii",
    "Canarina canariensis",
    "Cistus symphytifolius",
    "Erysimum scoparium",
    "Pericallis echinata",
    "Dracaena draco", # No es endémica, pero es un símbolo de las islas.
    "Echium virescens",
    "Plocama pendula",
    "Sonchus canariensis", # ¿Comparar ambos?
    "Sonchus acaulis"
]

print("Nº de plantas endémicas:", len(endemic_plants))

filtered_df = df[df["species"].isin(endemic_plants)]
print("Nº de imágenes de plantas endémicas seleccionadas:", len(filtered_df))

### Creación de scrapper ###
def clean_species_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()

def format_description(description):
    return description.split(":")[-1].strip()

class ImageFetcherSpider(Spider):
    name = "image-fetching"

    def __init__(self, df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
    
    def configure_sessions(self, manager):
        manager.add("default", AsyncStealthySession())

    async def start_requests(self):
        for index, row in self.df.iterrows():
            species = row["species"]
            image_url = row["imageUrl"]
            description = format_description(row["description"])

            file_name = f"{index:05d}.jpg" # Nombrado con el formato "nnnnn.jpg".
            folder_name = clean_species_name(species)

            full_path = f"data/raw/images/{folder_name}/{description}/{file_name}"
            yield Request(
                url=image_url,
                callback=self.parse,
                sid="default",
                meta={
                    "file_name": file_name,
                    "full_path": full_path
                }
            )

    async def parse(self, response: Response):
        if response.status == 200:
            metadata = response.meta
            file_name = metadata['file_name']
            full_path = metadata['full_path']

            image_bytes = response.body
            await self._save_file(image_bytes, full_path)
            
            print(f"Descargado {file_name}.")
            yield{"ok": True}
        else:
            print(f"Error descargando de {response.url}")
            yield{"ok": False}

    async def _save_file(self, image_bytes: bytes, full_path: str):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        async with aiofiles.open(full_path, "wb") as f:
            await f.write(image_bytes)

### Ejecución de scrapper (descarga de imágenes) ###
ImageFetcherSpider(filtered_df).start()
print("Descargas completadas.")