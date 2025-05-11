# Отслеживание носа и глаз

## Алгоритм действий

1. Клонировать репозиторий
```bash
git clone https://github.com/boraplyton/eyes-head
```
2. Создать и активировать виртуильное окружение
```bash
python -m venv venv
venv/Scripts/activate
```
3. Прописать 
```bash
pip install -r requirements.txt 
```
4. Запустить программу
```bash
python one_window.py
```


Все результаты сохраняются в папке `dataset`

Изменение/добавление квадратов происходит через файл `sq.txt`. Каждая строка файла - координаты левого верхнего угла отдельного квадрата (x, y)