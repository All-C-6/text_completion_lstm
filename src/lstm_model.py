import torch
import torch.nn as nn
import logging

from common_utils import setup_logging


setup_logging(log_file_name="lstm_model.log", level="INFO")
logger = logging.getLogger(__name__)


class LSTMNextTokenPredictor(nn.Module):
    """
    Модель для предсказания следующего токена или последовательности.

    Основная идея: LSTM с линейным выходом
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)  # Добавляем dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        logger.info(f"Модель LSTMNextTokenPredictor инициализирована с параметрами: vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_layers={dropout}")

    def forward(self, input_token_sequence, return_all_logits=False):
        """
        input_token_sequence: (batch_size, sequence_length) - последовательность токенов
        return_all_logits: bool - если True, возвращает логиты для всех позиций

        returns: 
            если return_all_logits=True: (batch_size, sequence_length, vocab_size)
            если return_all_logits=False: (batch_size, vocab_size) - только последний токен
        """
        # Embedding слой
        embedded_sequence = self.embedding(input_token_sequence)
        # (batch_size, sequence_length, embedding_dim)

        # LSTM слой
        lstm_output_sequence, (final_hidden_state, final_cell_state) = self.lstm(embedded_sequence)
        # lstm_output_sequence: (batch_size, sequence_length, hidden_dim)

        if return_all_logits:
            # Режим обучения: возвращаем логиты для всех позиций последовательности
            batch_size, sequence_length, hidden_dimension = lstm_output_sequence.shape

            # Reshape для применения линейного слоя ко всем hidden states
            lstm_output_reshaped = lstm_output_sequence.reshape(
                batch_size * sequence_length, 
                hidden_dimension
            )
            # (batch_size * sequence_length, hidden_dim)

            # Применяем линейный слой
            logits_for_all_positions = self.fc(lstm_output_reshaped)
            # (batch_size * sequence_length, vocab_size)

            # Reshape обратно в исходную форму
            logits_for_all_positions = logits_for_all_positions.view(
                batch_size, 
                sequence_length, 
                self.vocab_size
            )
            # (batch_size, sequence_length, vocab_size)

            return logits_for_all_positions
        else:
            # Режим генерации: возвращаем логиты только для последнего токена
            last_hidden_state = lstm_output_sequence[:, -1, :]
            # (batch_size, hidden_dim)

            logits_for_next_token = self.fc(last_hidden_state)
            # (batch_size, vocab_size)

            return logits_for_next_token

    def generate(self, initial_tokens, num_tokens_to_generate, temperature=1.0, top_k=None, top_p=None):
        """
        Генерирует продолжение последовательности токенов.

        Args:
            initial_tokens: (batch_size, seq_len) или (seq_len,) - начальная последовательность
            num_tokens_to_generate: int - сколько токенов сгенерировать
            temperature: float - температура для сэмплирования (выше = более случайно)
            top_k: int - ограничение на top-k сэмплирование (None = не использовать)
            top_p: float - nucleus sampling порог (None = не использовать)

        Returns:
            generated_sequence: (batch_size, seq_len + num_tokens_to_generate) - расширенная последовательность
        """
        self.eval()

        with torch.no_grad():
            # Если входная последовательность одномерная, добавляем batch dimension
            if initial_tokens.dim() == 1:
                initial_tokens = initial_tokens.unsqueeze(0)

            batch_size = initial_tokens.size(0)
            generated_sequence = initial_tokens.clone()

            # Инициализируем hidden и cell states
            hidden_state = None
            cell_state = None

            # Генерируем токены один за другим
            for step in range(num_tokens_to_generate):
                # На первом шаге обрабатываем всю начальную последовательность
                # На следующих шагах — только последний сгенерированный токен
                if step == 0:
                    input_sequence = generated_sequence
                else:
                    input_sequence = generated_sequence[:, -1:] # только последний токен

                # Embedding
                embedded = self.embedding(input_sequence)

                # LSTM с передачей hidden state
                if hidden_state is None:
                    lstm_output, (hidden_state, cell_state) = self.lstm(embedded)
                else:
                    lstm_output, (hidden_state, cell_state) = self.lstm(embedded, (hidden_state, cell_state))

                # Берем последний выход
                last_output = lstm_output[:, -1, :]

                # Получаем логиты
                logits = self.fc(last_output)  # (batch_size, vocab_size)

                # Применяем температуру
                logits = logits / temperature

                # Top-k sampling
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)

                # Top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Удаляем токены с cumulative probability > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Сдвигаем маску вправо, чтобы оставить хотя бы один токен
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Применяем softmax для получения вероятностей
                probabilities = torch.softmax(logits, dim=-1)

                # Добавляем небольшой шум для разнообразия
                probabilities = probabilities + 1e-8
                probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

                # Сэмплируем следующий токен
                next_token = torch.multinomial(probabilities, num_samples=1)

                # Добавляем сгенерированный токен к последовательности
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

            return generated_sequence
