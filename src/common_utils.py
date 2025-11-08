import logging
import os
from pathlib import Path
import requests


def setup_logging(log_file_name='ds.log', level="INFO"):

    script_dir = Path(__file__).parent.parent.absolute()
    log_file_path = script_dir / "logs" / log_file_name

    # Создаём директорию logs, если её нет (включая все родительские директории)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Проверка существования файла (если нужна)
    if not log_file_path.exists():
        log_file_path.touch()

    # Настраиваем логирование
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file_path,
        filemode='a',
        force=True
    )


logger = logging.getLogger(__name__)
setup_logging(log_file_name="utils.log")


def download_file(url, filename=None):
    """
    Скачивает файл по URL
    
    Args:
        url (str): URL файла для скачивания
        filename (str, optional): Имя для сохранения файла. Если не указано, 
                                будет использовано имя из URL
    
    Returns:
        str: Путь к сохраненному файлу
    """
    try:
        if os.path.exists(filename):
            print(f"Файл {filename} уже существует, пропускаем загрузку")
            logger.info(f"Файл {filename} уже существует, пропускаем загрузку")
            return filename
        logger.info(f"Начата загрузка файла {url}")
        # Отправляем GET-запрос
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Проверяем статус ответа
        
        # Определяем имя файла
        if filename is None:
            # Извлекаем имя файла из URL
            filename = url.split('/')[-1]
        
        # создание полного пути для файла
        script_dir = Path(__file__).parent.absolute()
        raw_file_path = script_dir.parent.absolute() / "data" / filename
        raw_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем файл
        with open(raw_file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        logger.info(f"Файл успешно скачан: {raw_file_path}")
        return filename
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при скачивании файла: {e}")
        return None

