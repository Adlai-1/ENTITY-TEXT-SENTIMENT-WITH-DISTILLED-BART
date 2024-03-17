# CLI codes are found here...
# Serves as the Frontend of our model... 

# imports
import click
import os
# just to get rid of unwanted warning and error logs...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0

import tensorflow as tf
import numpy as np

@click.command()
@click.option('--e', prompt="entity", 
              prompt_required=True, 
              help="Use this flag to input your text-entity.")

@click.option('--t', 
              prompt="text", 
              prompt_required=True,
              help="Use this flag to input your text for sentiment prediction.")

def hello(e, t):
    """
    CLI-General Info\n
    -------------------------------------------------\n
    (1) This is an entity-text sentiment prediction CLI.\n
    (2) The model used is pre-trained BART model (small bart).\n
    (3) Parameters: --e: text-entity, --t: text. Both are string values.\n
    (4) Both parameters are needed for the model to perform its task.\n
    (5) Expected Input format: 'Amazon I had a bad customer experience.'\n
    (6) With the example, Amazon is the text-entity (--e).\n
    (7) I had a bad customer experience is the text (--t).
    """
    
    # get user input
    input = " ".join([e, t])

    # run model to execute
    sentiment = model.predict([input])

    # get the index with the best probability
    senId = np.argmax(sentiment)

    # output conditions
    if senId == 0:
        click.echo({
            "sentiment": "Negative",
            "confidence": sentiment[0][senId]
        })
    
    elif senId == 1:
        click.echo({
            "sentiment": "Positive",
            "confidence": sentiment[0][senId]
        })
    
    elif senId == 2:
        click.echo({
            "sentiment": "Neutral",
            "confidence": sentiment[0][senId]
        })
    
    else:
        click.echo({
            "sentiment": "Irrelevant",
            "confidence": sentiment[0][senId]
        })
    
    
if __name__ == '__main__':
    # loader indicator
    with click.progressbar(range(10), fill_char="=", label='Loading Model...') as bar:
        for _ in bar:
            # Simulate loading process
            model = tf.keras.models.load_model("./sentiment-model.keras")
    
    # click-cli function
    hello()
