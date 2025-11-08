import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
from typing import List, Tuple, Dict

from common_utils import setup_logging


logger = logging.getLogger(__name__)
setup_logging(log_file_name="datasetizer.log", level="INFO")


class NextTokenPredictionDataset(Dataset):
    """
    Dataset для предсказания следующего токена на основе предыдущих.
    Преобразует текст в последовательности индексов токенов.
    """

    def __init__(
        self,
        csv_file_path: str,
        vocabulary_token_to_index: Dict[str, int],
        maximum_sequence_length: int,
        input_column_name: str = 'input',
        target_column_name: str = 'target'
    ):
        """
        Args:
            csv_file_path: путь к CSV файлу с данными
            vocabulary_token_to_index: словарь токен -> индекс
            maximum_sequence_length: максимальная длина последовательности
            input_column_name: название колонки с входными данными
            target_column_name: название колонки с целевыми данными
        """
        logger.info(f"Инициализация датасета из файла: {csv_file_path}")

        self.dataframe = pd.read_csv(csv_file_path)
        self.vocabulary_token_to_index = vocabulary_token_to_index
        self.maximum_sequence_length = maximum_sequence_length
        self.input_column_name = input_column_name
        self.target_column_name = target_column_name

        # Специальные токены
        self.padding_token = '<PAD>'
        self.unknown_token = '<UNK>'
        self.start_of_sequence_token = '<SOS>'
        self.end_of_sequence_token = '<EOS>'

        logger.info(f"Загружено {len(self.dataframe)} примеров из датасета")
        logger.info(f"Размер словаря: {len(self.vocabulary_token_to_index)} токенов")
        logger.info(f"Максимальная длина последовательности: {maximum_sequence_length}")

    def __len__(self) -> int:
        """Возвращает количество примеров в датасете"""
        return len(self.dataframe)

    def tokenize_text(self, text: str) -> List[str]:
        """
        Токенизирует текст на отдельные слова.

        Args:
            text: входной текст

        Returns:
            список токенов
        """
        # Простая токенизация по пробелам и пунктуации
        return text.lower().split()

    def convert_tokens_to_indices(self, token_list: List[str]) -> List[int]:
        """
        Конвертирует список токенов в список индексов.

        Args:
            token_list: список токенов

        Returns:
            список индексов
        """
        unknown_token_index = self.vocabulary_token_to_index.get(
            self.unknown_token, 0
        )

        indices = [
            self.vocabulary_token_to_index.get(token, unknown_token_index)
            for token in token_list
        ]

        return indices

    def pad_sequence(
        self,
        sequence_indices: List[int],
        target_length: int
    ) -> List[int]:
        """
        Дополняет последовательность паддингом до нужной длины.

        Args:
            sequence_indices: последовательность индексов
            target_length: целевая длина

        Returns:
            дополненная последовательность
        """
        padding_token_index = self.vocabulary_token_to_index.get(
            self.padding_token, 0
        )

        if len(sequence_indices) < target_length:
            padding_length = target_length - len(sequence_indices)
            sequence_indices = sequence_indices + [padding_token_index] * padding_length
        else:
            sequence_indices = sequence_indices[:target_length]

        return sequence_indices

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает один пример из датасета.

        Args:
            index: индекс примера

        Returns:
            (input_tensor, target_tensor)
        """
        row = self.dataframe.iloc[index]

        input_text = str(row[self.input_column_name])
        target_text = str(row[self.target_column_name])

        # Токенизация
        input_tokens = self.tokenize_text(input_text)
        target_tokens = self.tokenize_text(target_text)

        # Конвертация в индексы
        input_indices = self.convert_tokens_to_indices(input_tokens)
        target_indices = self.convert_tokens_to_indices(target_tokens)

        # Паддинг
        input_indices_padded = self.pad_sequence(
            input_indices,
            self.maximum_sequence_length
        )
        target_indices_padded = self.pad_sequence(
            target_indices,
            self.maximum_sequence_length
        )

        # Конвертация в тензоры
        input_tensor = torch.tensor(input_indices_padded, dtype=torch.long)
        target_tensor = torch.tensor(target_indices_padded, dtype=torch.long)

        return input_tensor, target_tensor


def build_vocabulary_from_dataframe(
    dataframe: pd.DataFrame,
    column_names: List[str],
    minimum_token_frequency: int = 2
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Строит словарь токенов из датафрейма.

    Args:
        dataframe: датафрейм с текстовыми данными
        column_names: названия колонок для анализа
        minimum_token_frequency: минимальная частота токена для включения в словарь

    Returns:
        (token_to_index, index_to_token) словари
    """
    logger.info("Начало построения словаря токенов")

    token_frequency_counter = {}

    # Подсчет частот токенов
    for column_name in column_names:
        for text in dataframe[column_name]:
            tokens = str(text).lower().split()
            for token in tokens:
                token_frequency_counter[token] = token_frequency_counter.get(token, 0) + 1

    logger.info(f"Найдено уникальных токенов: {len(token_frequency_counter)}")

    # Фильтрация по частоте
    filtered_tokens = [
        token for token, frequency in token_frequency_counter.items()
        if frequency >= minimum_token_frequency
    ]

    logger.info(
        f"После фильтрации (min_freq={minimum_token_frequency}): "
        f"{len(filtered_tokens)} токенов"
    )

    # Специальные токены
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

    # Построение словарей
    vocabulary_token_to_index = {token: idx for idx, token in enumerate(special_tokens)}

    for token in filtered_tokens:
        if token not in vocabulary_token_to_index:
            vocabulary_token_to_index[token] = len(vocabulary_token_to_index)

    vocabulary_index_to_token = {
        idx: token for token, idx in vocabulary_token_to_index.items()
    }

    logger.info(f"Итоговый размер словаря: {len(vocabulary_token_to_index)}")

    return vocabulary_token_to_index, vocabulary_index_to_token