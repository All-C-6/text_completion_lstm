import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer
from typing import Tuple
import logging

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
        maximum_sequence_length=512
    ):
        self.tokenizer = tokenizer
        self.maximum_sequence_length = maximum_sequence_length

        # Читаем все строки
        with open(file_path_to_text_data, 'r', encoding='utf-8') as file:
            self.lines_of_text = [line.strip() for line in file if line.strip()]

    def __len__(self):
        return len(self.lines_of_text)

    def __getitem__(self, index):
        text_line = self.lines_of_text[index]

        # Токенизируем с truncation и padding
        encoded_data = self.tokenizer(
            text_line,
            max_length=self.maximum_sequence_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded_data['input_ids'].squeeze(0)
        attention_mask = encoded_data['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def create_train_val_test_dataloaders_from_text_file(
    file_path_to_text_data: str,
    tokenizer = None,  # Изменил на None, чтобы создавать внутри
    maximum_sequence_length: int = 512,
    batch_size_for_training: int = 8,
    batch_size_for_validation: int = 16,
    batch_size_for_testing: int = 16,
    train_split_ratio: float = 0.8,
    validation_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    number_of_dataloader_workers: int = 2,
    random_seed_for_split: int = 42,
    shuffle_training_data: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт три DataLoader (train, validation, test) из текстового файла.

    Args:
        file_path_to_text_data: путь к txt файлу
        tokenizer: токенизатор (если None, создаётся GPT2Tokenizer)
        maximum_sequence_length: максимальная длина последовательности
        batch_size_for_training: размер батча для обучения
        batch_size_for_validation: размер батча для валидации
        batch_size_for_testing: размер батча для тестирования
        train_split_ratio: доля данных для обучения (0.8 = 80%)
        validation_split_ratio: доля данных для валидации (0.1 = 10%)
        test_split_ratio: доля данных для тестирования (0.1 = 10%)
        number_of_dataloader_workers: количество воркеров для DataLoader
        random_seed_for_split: seed для воспроизводимости разбиения
        shuffle_training_data: перемешивать ли данные при обучении

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader
    """
    # Проверяем, что соотношения в сумме дают 1.0
    total_split_ratio = train_split_ratio + validation_split_ratio + test_split_ratio
    assert abs(total_split_ratio - 1.0) < 1e-6, \
        f"Сумма train/val/test должна быть 1.0, получено: {total_split_ratio}"

    # Создаём токенизатор, если не передан
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    # Устанавливаем pad_token и padding_side для GPT моделей
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"pad_token не установлен, используется eos_token: '{tokenizer.eos_token}'")

    # ВАЖНО: устанавливаем left padding для decoder-only моделей
    tokenizer.padding_side = 'left'
    print(f"padding_side установлен в: '{tokenizer.padding_side}'")

    # Создаём collate_fn для правильного паддинга
    def collate_fn_with_left_padding(batch):
        """Collate функция с left padding для decoder-only моделей"""
        input_ids_list = [item['input_ids'] for item in batch]
        attention_mask_list = [item['attention_mask'] for item in batch]

        # Паддинг слева для input_ids
        input_ids_padded = pad_sequence(
            [torch.flip(ids, [0]) for ids in input_ids_list],  # Переворачиваем
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        input_ids_padded = torch.flip(input_ids_padded, [1])  # Переворачиваем обратно

        # Паддинг слева для attention_mask
        attention_mask_padded = pad_sequence(
            [torch.flip(mask, [0]) for mask in attention_mask_list],
            batch_first=True,
            padding_value=0
        )
        attention_mask_padded = torch.flip(attention_mask_padded, [1])

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded
        }

    # Создаём полный датасет
    full_dataset = TextFileLinesDataset(
        file_path_to_text_data=file_path_to_text_data,
        tokenizer=tokenizer,
        maximum_sequence_length=maximum_sequence_length
    )

    total_number_of_samples = len(full_dataset)
    print(f"Всего строк в датасете: {total_number_of_samples}")

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
        collate_fn=collate_fn_with_left_padding  # Добавляем collate_fn
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size_for_validation,
        shuffle=False,
        num_workers=number_of_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_left_padding  # Добавляем collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size_for_testing,
        shuffle=False,
        num_workers=number_of_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_left_padding  # Добавляем collate_fn
    )

    return train_dataloader, validation_dataloader, test_dataloader
