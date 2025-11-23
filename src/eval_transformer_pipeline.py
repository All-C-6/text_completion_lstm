import torch
from transformers import AutoModelForCausalLM
import evaluate


def evaluate_distilgpt2_rouge(tokenizer, gpt_model, validation_dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Оценивает модель distilgpt2 на валидационном датасете с помощью метрик ROUGE-1 и ROUGE-2.

    Args:
        tokenizer: токенизатор distilgpt2
        validation_dataloader: DataLoader с валидационными данными
        device: устройство для вычислений

    Returns:
        dict: словарь с метриками rouge1 и rouge2
    """
    gpt_model.to(device)
    # Загружаем модель
    gpt_model.eval()

    # Загружаем метрику ROUGE
    rouge = evaluate.load('rouge')

    predictions = []
    references = []

    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Берем 75% токенов для prompt
            prompt_length = int(input_ids.shape[1] * 0.75)
            prompt_ids = input_ids[:, :prompt_length]
            prompt_attention_mask = attention_mask[:, :prompt_length]

            # Генерируем оставшиеся 25%
            generated_ids = gpt_model.generate(
                prompt_ids,
                attention_mask=prompt_attention_mask,  # Добавьте attention_mask
                max_length=input_ids.shape[1], 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            # Декодируем предсказания и референсы
            print("Generated:")
            batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print("Reference:")
            batch_references = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            print("end")
            predictions.extend(batch_predictions)
            references.extend(batch_references)

    # Вычисляем ROUGE метрики
    results = rouge.compute(predictions=predictions, references=references)

    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2']
    }