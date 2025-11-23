import torch
import torch.nn as nn
from tqdm import tqdm
import evaluate
import logging
from common_utils import setup_logging


logger = logging.getLogger(__name__)
setup_logging(log_file_name="train.log", level="INFO")


def train_code_completion_model(
    model, 
    train_loader, 
    val_loader, 
    tokenizer, 
    n_epochs=10, 
    lr=0.001, 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    validate_every_n_epochs=2,
    num_val_samples_for_rouge=50,
    num_examples_to_display=5
):
    """
    Функция обучения модели автодополнения кода.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Загружаем ROUGE один раз
    rouge = evaluate.load('rouge')

    # Включаем оптимизации
    torch.backends.cudnn.benchmark = True

    model.to(device)

    for epoch in range(n_epochs):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
            # non_blocking=True для асинхронной передачи
            input_ids_batch = batch_data['input_ids'].to(device, non_blocking=True)
            attention_mask_batch = batch_data['attention_mask'].to(device, non_blocking=True)
            labels_batch = batch_data['labels'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids_batch, return_all_logits=True)

            # Reshape для loss
            batch_size, sequence_length, vocabulary_size = logits.shape
            logits_reshaped = logits.view(-1, vocabulary_size)
            labels_reshaped = labels_batch.view(-1)

            loss = criterion(logits_reshaped, labels_reshaped)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ==================== VALIDATION ====================
        # Быстрая валидация каждую эпоху (только loss)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]"):
                input_ids_batch = batch_data['input_ids'].to(device, non_blocking=True)
                labels_batch = batch_data['labels'].to(device, non_blocking=True)

                # Loss calculation
                logits = model(input_ids_batch, return_all_logits=True)

                batch_size, sequence_length, vocabulary_size = logits.shape
                logits_reshaped = logits.view(-1, vocabulary_size)
                labels_reshaped = labels_batch.view(-1)

                loss = criterion(logits_reshaped, labels_reshaped)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # ==================== ROUGE (раз в N эпох) ====================
        rouge1_score = None
        rouge2_score = None

        # Считаем ROUGE только раз в validate_every_n_epochs
        if (epoch + 1) % validate_every_n_epochs == 0:
            predictions = []
            references = []
            samples_processed = 0

            with torch.no_grad():
                for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [ROUGE]"):
                    if samples_processed >= num_val_samples_for_rouge:
                        break

                    input_ids_batch = batch_data['input_ids'].to(device, non_blocking=True)
                    attention_mask_batch = batch_data['attention_mask'].to(device, non_blocking=True)

                    actual_sequence_length = attention_mask_batch.sum(dim=1)

                    for sample_index in range(input_ids_batch.size(0)):
                        if samples_processed >= num_val_samples_for_rouge:
                            break

                        actual_length = int(actual_sequence_length[sample_index].item())
                        context_length = actual_length * 3 // 4
                        target_length = actual_length - context_length

                        if target_length <= 0:
                            continue

                        context_tokens = input_ids_batch[sample_index:sample_index+1, :context_length]

                        # Генерация
                        generated_tokens = model.generate(
                            context_tokens, 
                            target_length, 
                            temperature=1.0
                        )
                        generated_completion = generated_tokens[:, context_length:]

                        # Декодирование
                        predicted_text = tokenizer.decode(
                            generated_completion[0].cpu().tolist(),
                            skip_special_tokens=True
                        )
                        reference_text = tokenizer.decode(
                            input_ids_batch[sample_index, context_length:actual_length].cpu().tolist(),
                            skip_special_tokens=True
                        )

                        if predicted_text.strip() and reference_text.strip():
                            predictions.append(predicted_text)
                            references.append(reference_text)
                            samples_processed += 1

            # Вычисление ROUGE
            if predictions and references:
                rouge_results = rouge.compute(predictions=predictions, references=references)
                rouge1_score = rouge_results['rouge1']
                rouge2_score = rouge_results['rouge2']

        # ==================== ПРИМЕРЫ ДОПОЛНЕНИЯ ====================
        # Генерируем случайные примеры в конце каждой эпохи
        logger.info(f"\n{'='*80}")
        logger.info(f"Примеры автодополнения после эпохи {epoch+1}/{n_epochs}:")
        logger.info(f"{'='*80}\n")

        examples_displayed = 0

        with torch.no_grad():
            # Итерируемся по валидационному датасету и собираем примеры
            for batch_data in val_loader:
                if examples_displayed >= num_examples_to_display:
                    break

                input_ids_batch = batch_data['input_ids'].to(device, non_blocking=True)
                attention_mask_batch = batch_data['attention_mask'].to(device, non_blocking=True)

                actual_sequence_length = attention_mask_batch.sum(dim=1)
                batch_size = input_ids_batch.size(0)

                # Проходим по всем сэмплам в батче
                for sample_index in range(batch_size):
                    if examples_displayed >= num_examples_to_display:
                        break

                    actual_length = int(actual_sequence_length[sample_index].item())
                    context_length = actual_length * 3 // 4
                    target_length = actual_length - context_length

                    if target_length <= 0:
                        continue

                    context_tokens = input_ids_batch[sample_index:sample_index+1, :context_length]

                    # Генерация
                    generated_tokens = model.generate(
                        context_tokens, 
                        target_length, 
                        temperature=1.0
                    )
                    generated_completion = generated_tokens[:, context_length:]

                    # Декодирование
                    context_text = tokenizer.decode(
                        context_tokens[0].cpu().tolist(),
                        skip_special_tokens=True
                    )
                    predicted_text = tokenizer.decode(
                        generated_completion[0].cpu().tolist(),
                        skip_special_tokens=True
                    )
                    reference_text = tokenizer.decode(
                        input_ids_batch[sample_index, context_length:actual_length].cpu().tolist(),
                        skip_special_tokens=True
                    )

                    if predicted_text.strip() and reference_text.strip():
                        logger.info(f"Пример {examples_displayed + 1}:")
                        logger.info(f"Контекст: {context_text}")
                        logger.info(f"Предсказание: {predicted_text}")
                        logger.info(f"Референс: {reference_text}")
                        logger.info(f"{'-'*80}\n")
                        examples_displayed += 1

        if examples_displayed == 0:
            logger.info("Не удалось сгенерировать примеры (возможно, все последовательности слишком короткие)\n")

        # ==================== LOGGING ====================
        if rouge1_score is not None:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"ROUGE-1: {rouge1_score:.4f} | ROUGE-2: {rouge2_score:.4f}\n")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

    return model