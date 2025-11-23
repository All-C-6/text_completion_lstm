from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer
from typing import Tuple
import logging
import re
import os

from common_utils import setup_logging


logger = logging.getLogger(__name__)
setup_logging(log_file_name="data_process.log", level="INFO")

class TextFileLinesDataset(Dataset):
    """
    Датасет для загрузки текста построчно (каждая строка = один сэмпл).
    """
    def __init__(
        self, 
        file_path_to_text_data, 
        tokenizer, 
        maximum_sequence_length=512,
        maximum_number_of_rows=None,
        minimum_sequence_length=20  # НОВОЕ: минимальная длина последовательности
    ):
        self.tokenizer = tokenizer
        self.maximum_sequence_length = maximum_sequence_length
        self.minimum_sequence_length = minimum_sequence_length

        # Читаем все строки
        with open(file_path_to_text_data, 'r', encoding='utf-8') as file:
            all_lines = [line.strip() for line in file if line.strip()]

        # НОВОЕ: Фильтруем слишком короткие строки
        self.lines_of_text = []
        filtered_count = 0

        for line in all_lines:
            # Быстрая токенизация для проверки длины
            tokens = self.tokenizer.encode(line, add_special_tokens=True)
            if len(tokens) >= self.minimum_sequence_length:
                self.lines_of_text.append(line)
            else:
                filtered_count += 1

        logger.info(f"Отфильтровано {filtered_count} строк короче {self.minimum_sequence_length} токенов")
        logger.info(f"Осталось {len(self.lines_of_text)} валидных строк")

        # Урезаем датасет до указанного количества строк
        if maximum_number_of_rows is not None:
            self.lines_of_text = self.lines_of_text[:maximum_number_of_rows]

    def __len__(self):
        return len(self.lines_of_text)

    def __getitem__(self, index):
        text_line = self.lines_of_text[index]

        # Токенизируем БЕЗ padding (padding будет в collate_fn)
        encoded_data = self.tokenizer(
            text_line,
            max_length=self.maximum_sequence_length,
            truncation=True,
            padding=False,  # ИЗМЕНЕНО: убираем padding
            return_tensors='pt',
            add_special_tokens=True
        )

        input_ids = encoded_data['input_ids'].squeeze(0)
        attention_mask = encoded_data['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def clean_text_file(
    input_file_path: str,
    output_file_path: str,
    minimum_line_length: int = 20,
    encoding: str = 'utf-8'
) -> dict:
    """
    Очищает текстовый файл от мусора и сохраняет результат.

    Args:
        input_file_path: Путь к входному txt файлу
        output_file_path: Путь к выходному txt файлу
        minimum_line_length: Минимальная длина строки (по умолчанию 20 символов)
        encoding: Кодировка файлов (по умолчанию utf-8)

    Returns:
        dict: Статистика обработки (исходное кол-во строк, обработанное, удаленное)
    """

    # Регулярные выражения для очистки
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    mention_pattern = re.compile(r'@\w+')
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # эмодзи эмоций
        "\U0001F300-\U0001F5FF"  # символы и пиктограммы
        "\U0001F680-\U0001F6FF"  # транспорт и символы карт
        "\U0001F1E0-\U0001F1FF"  # флаги (iOS)
        "\U00002500-\U00002BEF"  # китайские символы
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )

    input_path = Path(input_file_path)
    output_path = Path(output_file_path)

    # Проверка существования входного файла
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_file_path}")

    # Создание директории для выходного файла, если не существует
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines_count = 0
    cleaned_lines_count = 0
    removed_lines_count = 0

    try:
        with open(input_path, 'r', encoding=encoding) as input_file, \
             open(output_path, 'w', encoding=encoding) as output_file:

            for line in input_file:
                total_lines_count += 1

                # Удаление пробелов по краям
                cleaned_line = line.strip()

                # Пропуск пустых строк
                if not cleaned_line:
                    removed_lines_count += 1
                    continue

                # Приведение к нижнему регистру
                cleaned_line = cleaned_line.lower()

                # Удаление ссылок
                cleaned_line = url_pattern.sub('', cleaned_line)

                # Удаление упоминаний @
                cleaned_line = mention_pattern.sub('', cleaned_line)

                # Удаление эмодзи
                cleaned_line = emoji_pattern.sub('', cleaned_line)

                # Замена нестандартных символов (оставляем буквы, цифры, основную пунктуацию)
                # Сохраняем русские и латинские буквы, цифры и базовые знаки пунктуации
                cleaned_line = re.sub(r'[^\w\s.,!?;:()\-—–«»""\'\d]', ' ', cleaned_line)

                # Замена множественных пробелов на один
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()

                # Проверка минимальной длины строки
                if len(cleaned_line) < minimum_line_length:
                    removed_lines_count += 1
                    continue

                # Запись очищенной строки
                output_file.write(cleaned_line + '\n')
                cleaned_lines_count += 1

        statistics = {
            'total_lines': total_lines_count,
            'cleaned_lines': cleaned_lines_count,
            'removed_lines': removed_lines_count,
            'removed_percentage': round(removed_lines_count / total_lines_count * 100, 2) if total_lines_count > 0 else 0
        }

        logger.info(f"Обработка txt завершена!")
        logger.info(f"Всего строк: {statistics['total_lines']}, удалено строк: {statistics['removed_lines']} ({statistics['removed_percentage']}%)")
        logger.info(f"Результат сохранён в: {output_file_path}")

        return statistics

    except Exception as exception:
        error_message = f"Ошибка при обработке файла: {str(exception)}"
        logger.error(error_message)
        raise


def create_train_val_test_dataloaders_from_text_file(
    file_path_to_text_data: str,
    tokenizer: GPT2Tokenizer,
    maximum_sequence_length: int = 512,
    minimum_sequence_length: int = 20,  # НОВОЕ: минимальная длина
    batch_size_for_training: int = 8,
    batch_size_for_validation: int = 16,
    batch_size_for_testing: int = 16,
    train_split_ratio: float = 0.8,
    validation_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    number_of_dataloader_workers: int = 2,
    random_seed_for_split: int = 42,
    shuffle_training_data: bool = True,
    max_rows_all: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт три DataLoader (train, validation, test) из текстового файла.
    """
    # предобработка файла
    filepath, ext = os.path.splitext(file_path_to_text_data)
    output_path = filepath + "_cleaned" + ext
    # если очищенный файл уже есть, то ничего не делаем
    if not os.path.exists(output_path):
        clean_text_file(file_path_to_text_data, output_path)

    # Проверяем, что соотношения в сумме дают 1.0
    total_split_ratio = train_split_ratio + validation_split_ratio + test_split_ratio
    assert abs(total_split_ratio - 1.0) < 1e-6, \
        f"Сумма train/val/test должна быть 1.0, получено: {total_split_ratio}"

    # Устанавливаем pad_token и padding_side для GPT моделей
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(f"pad_token не установлен, используется eos_token: '{tokenizer.eos_token}'")

    # Создаём collate_fn для правильного паддинга
    def collate_fn_with_left_padding(batch):
        """Collate функция с left padding для decoder-only моделей"""
        input_ids_list = [item['input_ids'] for item in batch]
        attention_mask_list = [item['attention_mask'] for item in batch]
        labels_list = [item['labels'] for item in batch]

        # Паддинг слева для input_ids
        input_ids_padded = pad_sequence(
            [torch.flip(ids, [0]) for ids in input_ids_list],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        input_ids_padded = torch.flip(input_ids_padded, [1])

        # Паддинг слева для attention_mask
        attention_mask_padded = pad_sequence(
            [torch.flip(mask, [0]) for mask in attention_mask_list],
            batch_first=True,
            padding_value=0
        )
        attention_mask_padded = torch.flip(attention_mask_padded, [1])

        # Паддинг слева для labels (с -100 для игнорирования в loss)
        labels_padded = pad_sequence(
            [torch.flip(lbl, [0]) for lbl in labels_list],
            batch_first=True,
            padding_value=-100  # -100 игнорируется в CrossEntropyLoss
        )
        labels_padded = torch.flip(labels_padded, [1])

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'labels': labels_padded
        }

    # Создаём полный датасет
    full_dataset = TextFileLinesDataset(
        file_path_to_text_data=output_path,
        tokenizer=tokenizer,
        maximum_sequence_length=maximum_sequence_length,
        minimum_sequence_length=minimum_sequence_length,  # ДОБАВЛЕНО
        maximum_number_of_rows=max_rows_all
    )

    total_number_of_samples = len(full_dataset)
    logger.debug(f"Всего валидных строк в датасете: {total_number_of_samples}")

    if total_number_of_samples == 0:
        raise ValueError("Датасет пуст после фильтрации! Проверьте файл данных.")

    # Вычисляем размеры для каждого split
    number_of_training_samples = int(total_number_of_samples * train_split_ratio)
    number_of_validation_samples = int(total_number_of_samples * validation_split_ratio)
    number_of_test_samples = total_number_of_samples - number_of_training_samples - number_of_validation_samples

    print(f"Train samples: {number_of_training_samples} ({train_split_ratio*100:.1f}%)")
    print(f"Validation samples: {number_of_validation_samples} ({validation_split_ratio*100:.1f}%)")
    print(f"Test samples: {number_of_test_samples} ({test_split_ratio*100:.1f}%)")

    # Устанавливаем seed для воспроизводимости
    generator_for_random_split = torch.Generator().manual_seed(random_seed_for_split)

    # Разбиваем датасет
    train_dataset, validation_dataset, test_dataset = random_split(
        full_dataset,
        [number_of_training_samples, number_of_validation_samples, number_of_test_samples],
        generator=generator_for_random_split
    )

    # Создаём DataLoader для каждого split с collate_fn
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_for_training,
        shuffle=shuffle_training_data,
        num_workers=number_of_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_left_padding
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size_for_validation,
        shuffle=False,
        num_workers=number_of_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_left_padding
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size_for_testing,
        shuffle=False,
        num_workers=number_of_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_left_padding
    )

    return train_dataloader, validation_dataloader, test_dataloader
