from io import BytesIO
from config import Config

from telegram import Update, Sticker
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
import logging
from translator import translator
from pokeye.pokeye import poke_predictor

# create config.py with your API_KEY
'''
import os

class Config(object):
    API_KEY = os.environ.get('API_KEY', '1234567890:XXXXXxxxxxxXXXXXxxX')
'''
API_KEY = Config.API_KEY

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def start(update: Update, _: CallbackContext) -> None:
    update.message.reply_text("Hi! Send me a Pokémon pic")


def get_help(update: Update, _: CallbackContext) -> None:
    update.message.reply_text("Send me a Pokémon pic")


def reply_message(update: Update, _: CallbackContext) -> None:
    m_id = update.message.message_id
    update.message.reply_text(translator(update.message.text), reply_to_message_id=m_id)


def photo(update: Update, context: CallbackContext) -> None:
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())

    res = poke_predictor(f)
    m_id = update.message.message_id
    context.bot.send_message(chat_id=update.message.chat_id, text=res, reply_to_message_id=m_id)


def sticker(update: Update, _: CallbackContext):
    update.message.reply_sticker(update.message.sticker)  # Возвращает отправленный стикер
    update.message.reply_text(update.message.sticker.emoji)  # Возвращает эмодзи на отправленный стикер


def emoji_to_sticker(update: Update, _: CallbackContext) -> None:
    update.message.reply_sticker(update.message.text)


def sticker_to_emoji(update: Update, _: CallbackContext) -> None:
    update.message.reply_text(update.message.sticker.emoji)  # Возвращает эмодзи на отправленный стикер


def main() -> None:
    updater = Updater(token=API_KEY, use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", get_help))
    # dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, reply_message))
    dispatcher.add_handler(MessageHandler(Filters.photo, photo))
    # dispatcher.add_handler(MessageHandler(Filters.sticker, sticker))
    # dispatcher.add_handler(MessageHandler(Filters.text, emoji_to_sticker))
    # dispatcher.add_handler(MessageHandler(Filters.sticker, sticker_to_emoji))

    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()
