import gc
import re

import torch
from transformers import EarlyStoppingCallback
from unsloth import FastVisionModel, train_on_responses_only
from unsloth.chat_templates import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split

from n1_dataset_formatting import build_dataset

def main():
    print("Cargando modelo...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "google/gemma-4-E2B",
        max_seq_length = 2048,
        full_finetuning = False,
        device_map = {"": 0}, # Fuerza todo a utilizar la GPU (de otra forma, estaba dando error por instrucciones en CPU).
        load_in_4bit = True # QLoRA.
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    model = FastVisionModel.get_peft_model(
        model,

        finetune_vision_layers = True, # Permite fine-tunear capas de visión.
        finetune_language_layers = True, # Permite fine-tunear capas de lenguaje.
        finetune_attention_modules = True, # Permite fine-tunear las capas de atención.
        finetune_mlp_modules = True,

        r = 16, # Establece el LoRA rank (tamaño de matriz de entrenamiento). r mayor = más uso de VRAM.
        lora_alpha = 16, # Establece la fuerza de las actualizaciones a los pesos. Se recomienda dejarlo al menos igual que el valor de r.
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",  # Optimizaciones de gradiente de Unsloth. Permiten guardar hasta un 30% de VRAM.
        random_state = 42
    )

    print("Modelo cargado correctamente. Cargando dataset...")
    dataset = build_dataset("data/raw/corpus", "data/raw/images/")
    if len(dataset) == 0:
      print("No se ha podido cargar el dataset.")
      return
    
    print("Dataset cargado correctamente. Preparando estratificación...")

    
    def extract_label(sample):
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                
                match = re.search(r"\*\*Species:\*\*\s+(.+?),", msg["content"][0]["text"])

                if match:
                    return match.group(1).strip()
    
    labels = [extract_label(sample) for sample in dataset]

    train_dataset, eval_dataset = train_test_split(
        dataset,
        stratify=labels,
        test_size=0.05,
        random_state=42
    )
    
    print("Dataset estratificado, cargando entrenador...")
    formatting_func = lambda conversation: conversation
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer.tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        formatting_func=formatting_func,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        args = SFTConfig(
            learning_rate = 2e-4,
            lr_scheduler_type = "cosine",
            optim = "adamw_8bit",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            num_train_epochs = 3,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            warmup_ratio = 0.05,
            max_grad_norm = 0.3,
            weight_decay = 0.001,

            output_dir = "./gemma4-plants-checkpoints",
            report_to = "none",
            eval_strategy="steps",
            save_strategy="steps",
            save_steps = 100,
            eval_steps = 20,
            dataloader_pin_memory = True,
            metric_for_best_model = "eval_loss",
            greater_is_better=False,

            # Parámetros obligatorios, según Unsloth.
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience = 5,
                early_stopping_threshold = 0.01,
            )
        ]
    )

    # Vacíamos cachés para tener toda la VRAM y RAM posible
    gc.collect()
    torch.cuda.empty_cache()

    print("Entrenador cargado, iniciando entrenamiento...")
    trainer.train()

if __name__ == "__main__":
    main()