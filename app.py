import gradio as gr

from utils import *


def classify(img):
    data = preprocess(img)
    pred = predict(data)
    return array_to_html_table(pred)


title = "Galaxy classifier"
description = "Classify a galaxy based on its properties."
article = """<p style='text-align: center'>
        <a href='https://nbviewer.org/github/kingjuno/Galaxy-Classification/blob/master/notebook/galaxy_zoo_checkpoint.ipynb?flush_cache=true' target='_blank'>Notebook Link</a>
        <br><a href='https://github.com/kingjuno/Galaxy-Classification' target='_blank'>Github Repo</a></p>
    """

examples = [
    ['samples/100008.jpg'],
    ['samples/956407.jpg'],
    ['samples/121006.jpg'],
    ['samples/563858.jpg'],
    ['samples/827985.jpg'],
]

gr.Interface(
    classify,
    title=title,
    description=description,
    article=article,
    examples=examples,
    inputs=gr.inputs.Image(type="filepath"),
    outputs="html",
).launch(debug=True, enable_queue=True)
