# Pokemon Predictor

1. clone repo with `git clone`
2. create and activate virtualenv
3. run `pip install -r requirements.txt`
4. create `config.py` with your API_KEY*
5. run `python bot.py`


> Telegram Bot: https://t.me/pokeye_bot

You can create your own models with `python pokeye/trainer.py`

#### *create config.py with your API_KEY in `Pokeye_bot` directory:
```
import os

class Config(object):
    API_KEY = os.environ.get('API_KEY', '1234567890:XXXXXxxxxxxXXXXXxxX')
```