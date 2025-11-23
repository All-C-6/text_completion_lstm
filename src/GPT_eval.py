import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rouge_score import rouge_scorer
import logging
import numpy as np

from common_utils import setup_logging


setup_logging(log_file_name="GPT_eval.log", level="INFO")
logger = logging.getLogger(__name__)


def validate_pretrained_gpt2_model(
    tokenizer,
    model,
    validation_dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_prediction_samples: int = 3,
    max_generation_length: int = 50,
    calculate_rouge_metrics: bool = True
) -> dict[str, float]:
    """
    Валидация предобученной модели GPT-2 без дообучения.

    Args:
        tokenizer: Токенизатор модели GPT-2
        model: Предобученная модель GPT2LMHeadModel
        validation_dataloader: DataLoader с валидационными данными
        device: Устройство для вычислений ('cuda' или 'cpu')
        num_prediction_samples: Количество примеров предсказаний для логирования
        max_generation_length: Максимальная длина генерируемого текста
        calculate_rouge_metrics: Вычислять ли ROUGE метрики (может быть медленным)

    Returns:
        Dict со значениями validation_loss, rouge1, rouge2
    """
    
    model.eval()
    model.to(device)

    total_validation_loss = 0.0
    total_batches = 0

    # Прогресс-бар для валидационных батчей
    validation_progress_bar = tqdm(
        validation_dataloader,
        desc="Validation",
        total=len(validation_dataloader),
        unit="it"
    )

    logger.info("Начинаем валидацию предобученной модели...")
    logger.info(f"Устройство: {device}")
    logger.info(f"Количество батчей: {len(validation_dataloader)}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Вычисление loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_validation_loss += loss.item()
            total_batches += 1

            # Обновление прогресс-бара
            validation_progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

    # Средний validation loss
    average_validation_loss = total_validation_loss / total_batches

    logger.info(f"\nValidation Loss: {average_validation_loss:.4f}")

    # Вычисление ROUGE метрик
    rouge1_score = 0.0
    rouge2_score = 0.0

    if calculate_rouge_metrics:
        logger.info("\nВычисляем ROUGE метрики...")
        rouge1_score, rouge2_score = calculate_rouge_scores_for_pretrained(
            model=model,
            tokenizer=tokenizer,
            validation_dataloader=validation_dataloader,
            device=device,
            num_prediction_samples=num_prediction_samples,
            max_generation_length=max_generation_length
        )

        logger.info(f"ROUGE-1: {rouge1_score:.4f}")
        logger.info(f"ROUGE-2: {rouge2_score:.4f}")

    # Итоговый вывод
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ПРЕДОБУЧЕННОЙ МОДЕЛИ")
    print("="*80)
    print(f"Validation Loss: {average_validation_loss:.4f}")
    if calculate_rouge_metrics:
        print(f"ROUGE-1: {rouge1_score:.4f}")
        print(f"ROUGE-2: {rouge2_score:.4f}")
    print("="*80 + "\n")

    return {
        'validation_loss': average_validation_loss,
        'rouge1': rouge1_score,
        'rouge2': rouge2_score
    }


def calculate_rouge_scores_for_pretrained(
    model,
    tokenizer,
    validation_dataloader: DataLoader,
    device: str,
    num_prediction_samples: int = 3,
    max_generation_length: int = 50
) -> tuple[float, float]:
    """
    Вычисление ROUGE-1 и ROUGE-2 метрик на валидационном наборе.

    Args:
        model: Предобученная модель GPT-2
        tokenizer: Токенизатор модели
        validation_dataloader: DataLoader с валидационными данными
        device: Устройство для вычислений
        num_prediction_samples: Количество примеров для логирования
        max_generation_length: Максимальная длина генерации

    Returns:
        Tuple с ROUGE-1 и ROUGE-2 скорами
    """
    logger = logging.getLogger(__name__)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []

    samples_logged = 0

    # Прогресс-бар для вычисления ROUGE
    rouge_progress_bar = tqdm(
        validation_dataloader,
        desc="ROUGE calculation",
        total=len(validation_dataloader),
        unit="it"
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(rouge_progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Определяем длину входной последовательности для генерации
            # Используем только часть input_ids как промпт
            prompt_length = input_ids.size(1) // 2  # Берем первую половину как промпт
            prompt_input_ids = input_ids[:, :prompt_length]
            prompt_attention_mask = attention_mask[:, :prompt_length]

            # Генерация предсказаний
            generated_outputs = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_length=prompt_length + max_generation_length,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # Декодирование предсказаний и референсов
            for i in range(input_ids.size(0)):
                # Предсказание (берем только сгенерированную часть)
                generated_text = tokenizer.decode(
                    generated_outputs[i][prompt_length:],
                    skip_special_tokens=True
                )

                # Референс (оригинальная целевая последовательность)
                reference_text = tokenizer.decode(
                    labels[i][labels[i] != -100],
                    skip_special_tokens=True
                )

                # Вычисление ROUGE
                if reference_text.strip() and generated_text.strip():
                    scores = scorer.score(reference_text, generated_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)

                # Логирование примеров предсказаний
                if samples_logged < num_prediction_samples:
                    prompt_text = tokenizer.decode(
                        prompt_input_ids[i],
                        skip_special_tokens=True
                    )
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Пример предсказания #{samples_logged + 1}:")
                    logger.info(f"Промпт: {prompt_text[:200]}...")
                    logger.info(f"Референс: {reference_text[:200]}...")
                    logger.info(f"Предсказание: {generated_text[:200]}...")
                    logger.info(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
                    logger.info(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
                    logger.info(f"{'='*80}\n")
                    samples_logged += 1

            # Обновление прогресс-бара со средними метриками
            if rouge1_scores:
                rouge_progress_bar.set_postfix({
                    'R1': f"{np.mean(rouge1_scores):.4f}",
                    'R2': f"{np.mean(rouge2_scores):.4f}"
                })

    # Средние значения ROUGE
    average_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
    average_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0.0

    return average_rouge1, average_rouge2
