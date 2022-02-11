import gradio as gr

from utils import *


def classify(img):
    data = preprocess(img)
    pred = predict(data)
    return array_to_html_table(pred)


title = "Galaxy classifier"
description = "Classify a galaxy based on its properties"

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
    examples=examples,
    inputs=gr.inputs.Image(type="filepath"),
    outputs="html",
).launch(debug=True, enable_queue=True)
