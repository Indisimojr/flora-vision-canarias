### Carga de librerías y datos ###
from scrapling.fetchers import AsyncStealthySession
from scrapling.spiders import Spider, Response, Request
import re
import os
import aiofiles

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

### Scrapper para búsqueda de páginas de información ###
# Carga todas las URLs, buscando en las páginas de "CanariWiki" y "EndemicasCanarias".
class CorpusSearchScrapper(Spider):
    name = "corpus-link-fetcher"

    def __init__(self, endemic_plants: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endemics_plants = endemic_plants

    def configure_sessions(self, manager):
        manager.add("default", AsyncStealthySession())
    
    async def start_requests(self):
        for plant_name in self.endemics_plants:
            yield Request(
                url=f"https://www3.gobiernodecanarias.org/medusa/wiki/index.php?search={plant_name.replace(' ', '+')}&title=Especial%3ABuscar&go=Ir",
                callback=self.parse_search_canariwiki,
                sid="default",
                meta={
                    "plant_name": plant_name
                }
            )

            yield Request(
                url=f"https://endemicascanarias.com/es/buscador?q={plant_name.replace(' ', '+')}",
                callback=self.parse_search_endemicascanarias,
                sid="default",
                meta={
                    "plant_name": plant_name
                }
            )

    async def parse(self, response: Response):
        pass

    async def parse_search_canariwiki(self, response: Response):
        plant_name = response.meta['plant_name']

        search_content = response.css("a.mw-redirect")
        for plant_page in search_content:
            text = plant_page.text
            href = plant_page.attrib.get("href", "")
            href = href.replace("&redirect=no", "") # Algunos resultados devuelven un parámetro "redirect=no" que debemos eliminar.

            if text and plant_name.lower() in text.lower():
                yield {"plant_name": plant_name, "source": "CanariWiki", "url": f"https://www3.gobiernodecanarias.org{href}"}
    
    async def parse_search_endemicascanarias(self, response: Response):
        plant_name = response.meta['plant_name']

        search_content = response.css("a.result__title-link")
        for plant_page in search_content:
            text = plant_page.css(".result__title-text::text").get()
            href = plant_page.attrib.get("href", "")
            if text and plant_name.lower() in text.lower():
                yield {"plant_name": plant_name, "source": "EndemicasCanarias", "url": f"https://endemicascanarias.com{href}"}

        yield {"plant_name": plant_name, "source": "EndemicasCanarias", "url": None}

corpus_search_results = CorpusSearchScrapper(endemic_plants).start()

corpus_urls = []
count_sources = { "CanariWiki": 0, "EndemicasCanarias": 0 }
for item in corpus_search_results:
    if not item['url']:
        continue
    
    corpus_urls.append(item)
    count_sources[item['source']] += 1 # Permite guardar la cantidad de documentos recuperados de cada fuente.

print("Recuento de documentos de cada origen: ", count_sources)

### Scrapper para recuperación de información de entradas de "CanariWiki" ###
def clean_species_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()

def add_markdown(type: str, content: str):
    if type == "header1":
        return f"# {content}\n\n"
    elif type == "header2":
        return f"## {content}\n\n"
    elif type == "paragraph":
        return f"{content}\n\n"

class CorpusContentScrapper(Spider):
    name = "corpus-fetching"

    def __init__(self, corpus_urls: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus_urls = corpus_urls

    def configure_sessions(self, manager):
        manager.add("default", AsyncStealthySession())
    
    async def start_requests(self):
        plants_count: dict = {}

        for corpus in self.corpus_urls:
            plant_name = corpus['plant_name']
            source = corpus['source']
            corpus_url = corpus['url']
            
            plant_count = plants_count.get(plant_name, 0) + 1 # Contamos el número de instancias de la planta. Inicia en 1, o añade 1 si ya ha sido inicializada.
            plants_count[plant_name] = plant_count

            file_name = f"{plant_count:05d}.md" # Ponemos un formato "nnnnn.md" para el archivo.
            full_path = f"data/raw/corpus/{clean_species_name(plant_name)}/{file_name}"

            if source == "CanariWiki":
                yield Request(
                    url=corpus_url,
                    callback=self.parse_canariwiki,
                    sid="default",
                    meta={
                        "full_path": full_path,
                        "plant_name": plant_name
                    }
                )
            else:
                yield Request(
                    url=corpus_url,
                    callback=self.parse_endemicascanarias,
                    sid="default",
                    meta={
                        "full_path": full_path,
                        "plant_name": plant_name
                    }
                )
    
    async def parse(self, response: Response):
        pass

    async def parse_canariwiki(self, response: Response):
        markdown_content = ""
        if response.status != 200:
            print(f"Error descargando de {response.url}")
            yield { "ok": False }

        full_path = response.meta['full_path']
        plant_name = response.meta['plant_name']

        main_header = response.css("h1#firstHeading")[0].css("::text").get()
        markdown_content += add_markdown("header1", main_header)

        main_content = response.css("div.mw-parser-output")[0]
        for element in main_content.css("h2, p"):
            if element.tag == "h2":
                text = element.css("span:last-child::text").get() # Tomamos todos los encabezados "h2" dentro de un "span". Ignoramos encabezados "invisibles" que salen como primer elemento.
                if text:
                    if text.lower() == "referencias" or "uso medicinal" in text.lower(): # Eliminamos las referencias, los últimos elementos que no nos interesan y apartados que no aportan información (como el de uso medicinal).
                        break

                    markdown_content += add_markdown("header2", text)

            elif element.tag == "p":
                if "foto" in element.text.getall().lower():
                    continue
                
                plant_names = element.css("span.comun, span.cientifico, span.cientifico > i")
                if plant_names.get():
                    markdown_content += f"Nombre común: *{plant_names[0].text.get()}*\n"
                    markdown_content += f"Nombre científico: *{plant_names[1].text.get()}*\n"
                    continue

                paragraph_content = element.css("::text").getall()
                text = ""
                for paragraph_text in paragraph_content:
                    text += paragraph_text.strip() + " "
                
                markdown_content += add_markdown("paragraph", text)
            
        await self._save_file(markdown_content, full_path)
        print(f"Descargado corpus de {plant_name} en CanariWiki.")
        yield { "ok": True }

    async def parse_endemicascanarias(self, response: Response):
        markdown_content = ""
        relevant_headers = [
            "Nombre común",
            "Familia",
            "Floración",
            "Distribución",
            "Características",
            "Hábitat y localización",
            "Usos"
        ]
        full_path = response.meta['full_path']
        plant_name = response.meta['plant_name']

        if response.status != 200:
            print(f"Error descargando de {response.url}")
            yield { "ok": False }
        main_content = response.css(".com-content-article__body")

        main_header = main_content.css("p a *::text").getall() # El encabezado es un "p" con una "a" dentro y varios elementos más ("em", "strong"...).
        main_header = " ".join(main_header) # Esto nos devuelve solo el texto ya correctamente.
        markdown_content += add_markdown("header1", main_header)

        for element in main_content.css("div > span, div > strong > span"):
            text = element.get_all_text(strip=True)

            if "referencias" in text.lower():
                break

            is_header = False
            for header in relevant_headers:
                if f"{header}:" in text: # Encabezado y contenido están juntos, así que los separamos.
                    paragraph = text.replace(f"{header}:", "")
                    paragraph = paragraph.replace("\n", " ").strip()
                        
                    markdown_content += add_markdown("header2", f"{header}:")
                    if paragraph != "":
                        markdown_content += add_markdown("paragraph", paragraph)

                    is_header = True
                    break
                
            if not is_header:
                markdown_content += add_markdown("paragraph", text)
            
        await self._save_file(markdown_content, full_path)
        print(f"Descargado corpus de {plant_name} en EndemicasCanarias.")
        yield {"ok": True}

    async def _save_file(self, markdown_content: str, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(markdown_content)

CorpusContentScrapper(corpus_urls).start()
print("Descarga completada.")