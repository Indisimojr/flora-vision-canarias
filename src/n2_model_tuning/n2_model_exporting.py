from pathlib import Path
import json

from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastVisionModel


PROJECT_ROOT = Path("E:/proyecto")

BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct"
BASE_TOKENIZER_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

ADAPTER_DIR = PROJECT_ROOT / "qwen3-vl-plants-lora" / "checkpoint-251"
HF_OUTPUT_DIR = PROJECT_ROOT / "qwen3-vl-plants-lora" / "qwen3-vl-plant-tuned"

MAX_SEQ_LENGTH = 4096


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def repair_qwen3_vl_metadata(output_dir: Path) -> None:
    """Restore metadata that LM Studio/GGUF conversion relies on."""
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_TOKENIZER_MODEL,
        trust_remote_code=True,
    )

    im_end_id = base_tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = base_tokenizer.convert_tokens_to_ids("<|endoftext|>")

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["chat_template"] = base_tokenizer.chat_template
    tokenizer_config["eos_token"] = "<|im_end|>"
    tokenizer_config["pad_token"] = "<|endoftext|>"
    tokenizer_config["bos_token"] = None
    write_json(tokenizer_config_path, tokenizer_config)

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["eos_token_id"] = im_end_id
    config["pad_token_id"] = endoftext_id
    config["bos_token_id"] = None
    write_json(config_path, config)

    generation_config_path = output_dir / "generation_config.json"
    if generation_config_path.exists():
        generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
    else:
        generation_config = {}

    generation_config["eos_token_id"] = [im_end_id, endoftext_id]
    generation_config["pad_token_id"] = endoftext_id
    generation_config["bos_token_id"] = endoftext_id
    write_json(generation_config_path, generation_config)

    print("Repaired Qwen3-VL metadata for GGUF/LM Studio.")
    print(f"chat_template present: {bool(tokenizer_config['chat_template'])}")
    print(f"eos_token_id: {generation_config['eos_token_id']}")
    print(f"pad_token_id: {generation_config['pad_token_id']}")


def main() -> None:
    model, processor = FastVisionModel.from_pretrained(
        BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )

    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model = model.merge_and_unload()

    HF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(HF_OUTPUT_DIR), safe_serialization=True)
    processor.save_pretrained(str(HF_OUTPUT_DIR))

    repair_qwen3_vl_metadata(HF_OUTPUT_DIR)
    print(f"Saved merged Hugging Face model to: {HF_OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# Exportar hacia GGUF.
# Ejecutar desde una carpeta de llama.cpp que tenga convert_hf_to_gguf.py disponible.
#
# py convert_hf_to_gguf.py E:\proyecto\qwen3-vl-plants-lora\qwen3-vl-plant-tuned --outfile E:\proyecto\qwen3-vl-plants-lora\qwen3-vl-plant-tuned.gguf --outtype q8_0
# py convert_hf_to_gguf.py E:\proyecto\qwen3-vl-plants-lora\qwen3-vl-plant-tuned --outfile E:\proyecto\qwen3-vl-plants-lora\mmproj-model.gguf --outtype q8_0 --mmproj
#
# LM Studio:
# - Cargar qwen3-vl-plant-tuned.gguf como modelo principal.
# - Cargar mmproj-model.gguf como proyector de visión si LM Studio no lo detecta automáticamente.
# - Stop strings recomendados: <|im_end|>, <|endoftext|>
# - Max tokens recomendado para esta tarea: 32-64
