import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from typing import Dict, List
import logging
from tqdm import tqdm

from lstm_model import LSTMTextPredictionModel

logger = logging.getLogger(__name__)


def convert_token_indices_to_text(
    token_indices: torch.Tensor,
    index_to_token_vocabulary: Dict[int, str],
    skip_special_tokens: bool = True
) -> str:
    """
    Конвертирует тензор индексов токенов в текстовую строку.

    Args:
        token_indices: тензор с индексами токенов
        index_to_token_vocabulary: словарь индекс -> токен
        skip_special_tokens: пропускать ли специальные токены

    Returns:
        текстовая строка
    """
    special_tokens = {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}

    tokens = []
    for token_index in token_indices.cpu().numpy():
        token = index_to_token_vocabulary.get(int(token_index), '<UNK>')

        if skip_special_tokens and token in special_tokens:
            continue

        tokens.append(token)

    return ' '.join(tokens)


def generate_text_completion_from_prefix(
    model: LSTMTextPredictionModel,
    input_sequence: torch.Tensor,
    fraction_of_input_to_use: float,
    number_of_tokens_to_generate: int,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95
) -> torch.Tensor:
    """
    Генерирует дополнение текста на основе части входной последовательности.

    Args:
        model: обученная LSTM модель
        input_sequence: входная последовательность [batch_size, sequence_length]
        fraction_of_input_to_use: какая часть входа используется (например, 0.75 = 3/4)
        number_of_tokens_to_generate: количество токенов для генерации
        device: устройство
        temperature: температура для сэмплирования
        top_k: top-k фильтрация
        top_p: nucleus sampling

    Returns:
        сгенерированная последовательность токенов
    """
    model.eval()

    sequence_length = input_sequence.size(1)
    prefix_length = int(sequence_length * fraction_of_input_to_use)

    input_prefix = input_sequence[:, :prefix_length].to(device)

    with torch.no_grad():
        generated_tokens = model.generate_next_tokens(
            initial_token_indices=input_prefix,
            number_of_tokens_to_generate=number_of_tokens_to_generate,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    return generated_tokens


def calculate_rouge_metrics(
    predicted_texts: List[str],
    reference_texts: List[str]
) -> Dict[str, float]:
    """
    Вычисляет метрики ROUGE для списка предсказаний.

    Args:
        predicted_texts: список предсказанных текстов
        reference_texts: список эталонных текстов

    Returns:
        словарь с метриками ROUGE
    """
    logger.info(f"Вычисление ROUGE метрик для {len(predicted_texts)} примеров")

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for predicted, reference in zip(predicted_texts, reference_texts):
        if not predicted.strip() or not reference.strip():
            continue

        scores = scorer.score(reference, predicted)

        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    average_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0
    average_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0.0
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    metrics = {
        'rouge-1': average_rouge_1,
        'rouge-2': average_rouge_2,
        'rouge-l': average_rouge_l
    }

    logger.info("ROUGE метрики вычислены:")
    logger.info(f"  ROUGE-1: {average_rouge_1:.4f}")
    logger.info(f"  ROUGE-2: {average_rouge_2:.4f}")
    logger.info(f"  ROUGE-L: {average_rouge_l:.4f}")

    return metrics


def evaluate_model_with_rouge(
    model: LSTMTextPredictionModel,
    data_loader: DataLoader,
    device: torch.device,
    index_to_token_vocabulary: Dict[int, str],
    number_of_samples_to_evaluate: int = 100,
    fraction_of_input_to_use: float = 0.75
) -> Dict[str, float]:
    """
    Оценивает модель с использованием метрик ROUGE.

    Args:
        model: модель для оценки
        data_loader: загрузчик данных
        device: устройство
        index_to_token_vocabulary: словарь индекс -> токен
        number_of_samples_to_evaluate: количество примеров для оценки
        fraction_of_input_to_use: какая часть входа используется

    Returns:
        словарь с метриками ROUGE
    """
    logger.info(
        f"Оценка модели с использованием ROUGE на {number_of_samples_to_evaluate} примерах"
    )

    model.eval()

    predicted_texts_list = []
    reference_texts_list = []

    samples_processed = 0

    with torch.no_grad():
        for input_batch, target_batch in tqdm(data_loader, desc="Оценка модели"):
            if samples_processed >= number_of_samples_to_evaluate:
                break

            input_batch = input_batch.to(device)
            batch_size = input_batch.size(0)
            sequence_length = input_batch.size(1)

            # Используем fraction_of_input_to_use часть входа
            prefix_length = int(sequence_length * fraction_of_input_to_use)
            number_of_tokens_to_generate = sequence_length - prefix_length

            # Генерируем дополнения для всего батча
            for sample_index in range(batch_size):
                if samples_processed >= number_of_samples_to_evaluate:
                    break

                single_input = input_batch[sample_index:sample_index + 1]
                single_target = target_batch[sample_index]

                # Генерируем текст
                generated_tokens = generate_text_completion_from_prefix(
                    model=model,
                    input_sequence=single_input,
                    fraction_of_input_to_use=fraction_of_input_to_use,
                    number_of_tokens_to_generate=number_of_tokens_to_generate,
                    device=device,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )

                # Конвертируем в текст
                predicted_text = convert_token_indices_to_text(
                    token_indices=generated_tokens[0],
                    index_to_token_vocabulary=index_to_token_vocabulary,
                    skip_special_tokens=True
                )

                # Получаем эталонный текст (последняя часть целевой последовательности)
                reference_tokens = single_target[prefix_length:]
                reference_text = convert_token_indices_to_text(
                    token_indices=reference_tokens,
                    index_to_token_vocabulary=index_to_token_vocabulary,
                    skip_special_tokens=True
                )

                if predicted_text.strip() and reference_text.strip():
                    predicted_texts_list.append(predicted_text)
                    reference_texts_list.append(reference_text)
                    samples_processed += 1

    logger.info(f"Обработано {samples_processed} примеров для оценки")

    # Вычисляем ROUGE метрики
    rouge_metrics = calculate_rouge_metrics(
        predicted_texts=predicted_texts_list,
        reference_texts=reference_texts_list
    )

    return rouge_metrics


def evaluate_and_display_examples(
    model: LSTMTextPredictionModel,
    data_loader: DataLoader,
    device: torch.device,
    index_to_token_vocabulary: Dict[int, str],
    number_of_examples_to_display: int = 10
):
    """
    Оценивает модель и отображает примеры предсказаний.

    Args:
        model: модель для оценки
        data_loader: загрузчик данных
        device: устройство
        index_to_token_vocabulary: словарь индекс -> токен
        number_of_examples_to_display: количество примеров для отображения
    """
    logger.info("=" * 80)
    logger.info("ДЕМОНСТРАЦИЯ ПРИМЕРОВ ПРЕДСКАЗАНИЙ МОДЕЛИ")
    logger.info("=" * 80)

    model.eval()

    examples_shown = 0

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            if examples_shown >= number_of_examples_to_display:
                break

            input_batch = input_batch.to(device)
            batch_size = input_batch.size(0)
            sequence_length = input_batch.size(1)

            prefix_length = int(sequence_length * 0.75)
            number_of_tokens_to_generate = sequence_length - prefix_length

            for sample_index in range(batch_size):
                if examples_shown >= number_of_examples_to_display:
                    break

                single_input = input_batch[sample_index:sample_index + 1]
                single_target = target_batch[sample_index]

                # Входной префикс
                input_prefix = single_input[0, :prefix_length]
                input_text = convert_token_indices_to_text(
                    token_indices=input_prefix,
                    index_to_token_vocabulary=index_to_token_vocabulary
                )

                # Целевое дополнение
                target_suffix = single_target[prefix_length:]
                target_text = convert_token_indices_to_text(
                    token_indices=target_suffix,
                    index_to_token_vocabulary=index_to_token_vocabulary
                )

                # Сгенерированное дополнение
                generated_tokens = generate_text_completion_from_prefix(
                    model=model,
                    input_sequence=single_input,
                    fraction_of_input_to_use=0.75,
                    number_of_tokens_to_generate=number_of_tokens_to_generate,
                    device=device
                )

                predicted_text = convert_token_indices_to_text(
                    token_indices=generated_tokens[0],
                    index_to_token_vocabulary=index_to_token_vocabulary
                )

                logger.info(f"\n{'='*60}")
                logger.info(f"Пример {examples_shown + 1}:")
                logger.info(f"{'='*60}")
                logger.info(f"Входной контекст (75%):\n  {input_text}")
                logger.info(f"\nЦелевое дополнение:\n  {target_text}")
                logger.info(f"\nСгенерированное дополнение:\n  {predicted_text}")
                logger.info(f"{'='*60}\n")

                examples_shown += 1

    logger.info("Демонстрация примеров завершена\n")