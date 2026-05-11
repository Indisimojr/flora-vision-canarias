import gc
import torch
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from n1_dataset_formatting import build_dataset

def main():
    print("Cargando modelo...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "google/gemma-4-E2B",
        max_seq_length = 1024,
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
        finetune_mlp_modules = False,

        r = 16, # Establece el LoRA rank (tamaño de matriz de entrenamiento). r mayor = más uso de VRAM.
        lora_alpha = 16, # Establece la fuerza de las actualizaciones a los pesos. Se recomienda dejarlo al menos igual que el valor de r.
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",  # Optimizaciones de gradiente de Unsloth. Permiten guardar hasta un 30% de VRAM.
        random_state = 42
    )

    print("Modelo cargado correctamente. Cargando dataset...")
    dataset = build_dataset("data/raw/images/")
    if len(dataset) == 0:
      print("No se ha podido cargar el dataset.")
      return
    print("Dataset cargado correctamente.")

    print("Cargando entrenador...")
    formatting_func = lambda conversation: conversation # Parámetro necesario para el funcionamiento (como ya tiene el formato correcto, no es necesario que haga nada).
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        learning_rate = 2e-4,
        formatting_func=formatting_func,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            save_steps = 200,
            num_train_epochs = 10,
            optim = "adamw_8bit",
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            warmup_ratio = 0.03,
            lr_scheduler_type = "cosine",
            output_dir = "./gemma4-plants-checkpoints",
            report_to = "none",
            dataloader_pin_memory = True,

            # Parámetros obligatorios, según Unsloth.
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 1024,
        ),
    )
    
    if torch.cuda.get_device_capability()[0] >= 8:  # Si tenemos por encima de cierta versión de GPUs, podemos recompilar para ahorrar algo de VRAM.
        model = torch.compile(model, mode="reduce-overhead")

    # Vacíamos cachés para tener toda la VRAM y RAM posible
    gc.collect()
    torch.cuda.empty_cache()

    print("Entrenador cargado, iniciando entrenamiento...")
    trainer.train()

if __name__ == "__main__":
    main()