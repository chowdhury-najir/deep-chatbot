# Build your own deep learning chatbot

This is my version of tensorflow's NMT (Neural Machine Translation) for chatbot using my own custom based LanguageBase (NOTE: This uses Python 3.6)

# Requirements

- tensorflow == 1.4.1
- nltk
- numpy
- sklearn

# Usage

1. Clone the repository via `git clone https://github.com/noahchalifour/deep-chatbot`
2. Change into the directory `cd deep-chatbot`
3. Install the required packages `pip install -r requirements.txt`
4. (Optional) Replace data in the `data` directory with your own custom data
5. Create a model directory `mkdir model`
6. Tune your model via the `params.py` file
7. Once you are ready you can train your model via `python train.py`
