from unsloth import FastVisionModel
from peft import PeftModel

# Cargamos el modelo fine-tuneado.
model, tokenizer = FastVisionModel.from_pretrained(
    "gemma4-plants-checkpoints/checkpoint-2950",
    load_in_4bit = True,
)

# Exportamos el modelo con la misma cuantización inicial.
model.save_pretrained_gguf(
    "gemma4-plants-checkpoints/gemma4-flower-tuned",
    tokenizer,
    quantization_method = "q4_k_m"
)