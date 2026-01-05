# 🧠 Neural Chatbot with PyTorch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## 🇬🇧 English Description

A fully functional, customizable chatbot built from scratch using **Python** and **PyTorch**.
Unlike simple rule-based bots, this project uses a **Feed Forward Neural Network** to classify user intents based on natural language patterns.

It does **not** rely on heavy pre-trained models or external APIs. It implements the "Bag of Words" technique and a custom neural architecture manually to demonstrate the fundamentals of NLP and Deep Learning.

### 📂 Project Structure
* `train.py` - Script to train the neural network.
* `chat.py` - The inference script to chat with the bot.
* `model.py` - The PyTorch neural network architecture (Feed Forward).
* `nltk_utils.py` - Helper functions for tokenization and stemming.
* `intents.json` - The training dataset (intents, patterns, and responses).

### 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ENC3LL/SimpleNeuralChatbot.git](https://github.com/ENC3LL/SimpleNeuralChatbot.git)
    cd SimpleNeuralChatbot
    ```

2.  **Install dependencies:**
    You need PyTorch, NLTK, and NumPy.
    ```bash
    pip install torch nltk numpy
    ```

3.  **Download NLTK data:**
    You might need to download the tokenizer data inside a python shell:
    ```python
    import nltk
    nltk.download('punkt')
    ```

### 🛠 Usage

**Step 1: Train the Model**
Before chatting, you need to train the neural network on the `intents.json` file.
```bash
python train.py

```

*This will generate a `data.pth` file containing the trained model weights.*

**Step 2: Chat**
Run the chatbot script to start a conversation.

```bash
python chat.py

```

### ⚙️ Customization

To teach the bot new phrases, simply edit `intents.json`.

```json
{
  "tag": "weather",
  "patterns": ["Is it raining?", "What's the weather?"],
  "responses": ["I am a bot, look out the window!", "It is sunny mostly."]
}

```

After editing the JSON, **run `train.py` again** to update the model.

---

<a name="russian"></a>

## 🇷🇺 Описание на Русском

Полностью рабочий, настраиваемый чат-бот, написанный с нуля на **Python** и **PyTorch**.
В отличие от простых ботов на `if/else`, этот проект использует **нейросеть прямого распространения (Feed Forward NN)** для классификации намерений пользователя.

Проект **не использует** тяжелые предобученные модели или внешние API. Здесь вручную реализована техника "Мешок слов" (Bag of Words) и архитектура нейросети, что отлично подходит для изучения основ NLP и Deep Learning.

### 📂 Структура проекта

* `train.py` - Скрипт для обучения нейросети.
* `chat.py` - Скрипт для запуска чата (инференс).
* `model.py` - Архитектура нейросети на PyTorch.
* `nltk_utils.py` - Утилиты для обработки текста (токенизация, стемминг).
* `intents.json` - Датасет для обучения (намерения, фразы и ответы).

### 🚀 Установка

1. **Клонируйте репозиторий:**
```bash
git clone [https://github.com/ENC3LL/SimpleNeuralChatbot.git](https://github.com/ENC3LL/SimpleNeuralChatbot.git)
cd SimpleNeuralChatbot

```


2. **Установите зависимости:**
Вам понадобятся PyTorch, NLTK и NumPy.
```bash
pip install torch nltk numpy

```


3. **Загрузите данные NLTK:**
Возможно, потребуется скачать токенизатор через Python консоль:
```python
import nltk
nltk.download('punkt')

```



### 🛠 Использование

**Шаг 1: Обучение модели**
Перед началом общения нужно обучить нейросеть на данных из `intents.json`.

```bash
python train.py

```

*После завершения появится файл `data.pth` с весами обученной модели.*

**Шаг 2: Чат**
Запустите скрипт чата для начала разговора.

```bash
python chat.py

```

### ⚙️ Настройка (Кастомизация)

Чтобы научить бота новым фразам, просто отредактируйте файл `intents.json`.

```json
{
  "tag": "погода",
  "patterns": ["Идет ли дождь?", "Какая сейчас погода?"],
  "responses": ["Я всего лишь бот, посмотри в окно!", "Кажется, солнечно."]
}

```

После изменения JSON файла **обязательно запустите `train.py` снова**, чтобы обновить модель.

```

```
