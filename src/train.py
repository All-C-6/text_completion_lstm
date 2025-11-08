import torch
import torch.nn as nn
from tqdm import tqdm
import evaluate
import torch.optim as optim
import logging
from common_utils import setup_logging


logger = logging.getLogger(__name__)
setup_logging(log_file_name="train.log", level="INFO")


def train_code_completion_model(model, train_loader, val_loader, tokenizer, n_epochs=10, lr=0.001, device='cpu'):
    """
    Функция обучения модели автодополнения кода с валидацией по ROUGE-1 и ROUGE-2.

    Args:
        model: модель LSTM
        train_loader: DataLoader для тренировочных данных
        val_loader: DataLoader для валидационных данных
        tokenizer: токенизатор для декодирования предсказаний
        n_epochs: количество эпох тренировки
        lr: гиперпараметр learning rate
        device: устройство для обучения ('cpu' по-умолчанию или 'cuda')
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    rouge = evaluate.load('rouge')
    model.to(device)

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.
        predictions = []
        references = []

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]"):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Loss calculation
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                # Generate completion (1/4 of sequence length)
                context_length = x_batch.size(1) * 3 // 4
                context = x_batch[:, :context_length]
                target_length = x_batch.size(1) - context_length

                generated = model.generate(context, target_length, temperature=1.0)
                generated_completion = generated[:, context_length:]

                # Decode for ROUGE
                for i in range(generated_completion.size(0)):
                    pred_text = tokenizer.decode(generated_completion[i].cpu().tolist())
                    ref_text = tokenizer.decode(x_batch[i, context_length:].cpu().tolist())
                    predictions.append(pred_text)
                    references.append(ref_text)

        val_loss /= len(val_loader)

        # ROUGE metrics
        rouge_results = rouge.compute(predictions=predictions, references=references)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | "
              f"ROUGE-1: {rouge_results['rouge1']:.4f} | ROUGE-2: {rouge_results['rouge2']:.4f}")

    return model