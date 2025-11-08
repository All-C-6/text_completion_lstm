import torch
import torch.nn as nn


class LSTMNextTokenPredictor(nn.Module):
    """
    Модель для предсказания следующего токена или последовательности.
    
    Основная идея: LSTM с линейным выходом
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len) - последовательность токенов
        returns: (batch_size, vocab_size) - logits для следующего токена
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        logits = self.fc(last_hidden)  # (batch_size, vocab_size)
        return logits

    def generate(self, initial_tokens, num_tokens_to_generate, temperature=1.0):
        """
        initial_tokens: (batch_size, seq_len) или (seq_len,) - начальная последовательность
        num_tokens_to_generate: int - сколько токенов сгенерировать
        temperature: float - температура для сэмплирования (выше = более случайно)
        returns: (batch_size, seq_len + num_tokens_to_generate) - расширенная последовательность
        """
        self.eval()
        with torch.no_grad():
            if initial_tokens.dim() == 1:
                initial_tokens = initial_tokens.unsqueeze(0)

            generated_sequence = initial_tokens.clone()

            for _ in range(num_tokens_to_generate):
                logits = self.forward(generated_sequence)  # (batch_size, vocab_size)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

            return generated_sequence


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())