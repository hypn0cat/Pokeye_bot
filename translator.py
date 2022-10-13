from textblob import TextBlob


def translator(phrase):
    try:
        lang = TextBlob(phrase).detect_language()
        ru_phrase = TextBlob(phrase).translate(to="ru")
        print(phrase, ru_phrase, sep=" --- ")
        return f'Translating {lang}->ru: {ru_phrase}'
    except Exception as e:
        print("Error:", e)
        return "Can't translate"

