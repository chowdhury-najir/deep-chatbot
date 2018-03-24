# Build your own deep learning chatbot

This is my version of tensorflow's NMT (Neural Machine Translation) for chatbot using my own custom based LanguageBase (NOTE: This uses Python 3.6)

# Requirements

<ul>
  <li>tensorflow==1.4.1</li>
  <li>nltk</li>
  <li>numpy</li>
  <li>sklearn</li>
</ul>

# Usage

<ol>
  <li>Clone the repository via `git clone https://github.com/noahchalifour/deep-chatbot`</li>
  <li>Change into the directory `cd deep-chatbot`</li>
  <li>Install the required packages `pip install -r requirements.txt`</li>
  <li>(Optional) Replace data in the `data` directory with your own custom data</li>
  <li>Create a model directory `mkdir model`</li>
  <li>Tune your model via the `params.py` file</li>
  <li>Once you are ready you can train your model via `python train.py`</li>
</ol>
