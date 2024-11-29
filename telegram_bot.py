from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

MODEL_NAME = "distilbert-base-multilingual-cased"
MODEL_PATH = "best_model_distilbert.pth"

# Classe do modelo usada durante o treinamento
class TextClassifier(nn.Module):
    def __init__(self, bert_model):
        super(TextClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

# Inicializa o modelo e o tokenizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModel.from_pretrained(MODEL_NAME)
model = TextClassifier(base_model).to(device)

# Carregar os pesos do modelo salvo
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar os pesos: {e}")

model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Função para processar e classificar texto
def classify_text(text):
    # Tokenizar o texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    # Fazer a predição
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs, dim=1).item()

    # Mapeamento da classe prevista
    class_name = "Poema" if predicted_class == 0 else "Crítica"
    return class_name

# Função de resposta para o comando /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Olá! Envie-me um texto para que eu possa classificá-lo como Poema ou Crítica.")

# Função para lidar com mensagens de texto
async def handle_text(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text

    # Classificar o texto
    class_name = classify_text(user_text)
    await update.message.reply_text(f"Classe prevista: {class_name}")

# Configuração do bot com o token do BotFather
def main():
    application = Application.builder().token("TOKEN DO BOT").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling()

if __name__ == '__main__':
    main()
