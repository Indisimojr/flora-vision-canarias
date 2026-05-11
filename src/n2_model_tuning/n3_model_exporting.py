from unsloth import FastVisionModel
from peft import PeftModel

# 1. Load the base model
model, tokenizer = FastVisionModel.from_pretrained(
    "gemma4-plants-checkpoints/checkpoint-2950",
    load_in_4bit = True,
)

# 3. Export
model.save_pretrained_gguf(
    "gemma4-plants-checkpoints/gemma4-flower-tuned",
    tokenizer,
    quantization_method = "q8_0"
)