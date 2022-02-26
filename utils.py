import albumentations as A
import cv2
import numpy as np
import onnxruntime


def preprocess(img_filepath):
    image = cv2.imread(img_filepath)
    image = (np.array(image).astype(np.float32))/255.

    test_transform = A.Compose([
        A.Resize(width=424, height=424, p=1.0),
        A.Crop(
            x_min=(424-256)//2,
            y_min=(424-256)//2,
            x_max=(424-256)//2+256,
            y_max=(424-256)//2+256,
        ),
        A.Resize(width=64, height=64, p=1.0)
    ])

    aug = test_transform(image=image)
    image = aug['image']
    image = image.transpose((2, 0, 1))

    image = np.stack([image]*1)
    print(image, image.shape)
    return image


def predict(input_img):
    model_onnx = 'onnx/galaxy_classification.onnx'

    session = onnxruntime.InferenceSession(model_onnx, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_img})

    return np.array(result)


def array_to_html_table(array):
    array = array[0].reshape(1, -1)
    html_table = """
    <style>
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }
        th, td {
            padding: 5px;
            text-align: left;
        }
    </style>
    """
    columns_heading = [
        "Smooth",
        "features or disk",
        "star or artifact",
        "Disk Viewed edge on (Yes)",
        "Disk Viewed edge on (No)",
        "Bar feature through center of galaxy (Yes)",
        "Bar feature through center of galaxy (No)",
        "Spiral arms (Yes)",
        "Spiral arms (No)",
        "Bulge (No)",
        "Bulge (Noticable)",
        "Bulge (Obvious)",
        "Bulge (Prominent)",
        "Odd features (Yes)",
        "Odd features (No)",
        "Completely Round (Yes)",
        "Completely Round (in between)",
        "cigar shape (Yes)",
        "Ring",
        "Lens or arc",
        "Disturbed",
        "Irregular",
        "Other shape",
        "merger",
        "dust lane",
        "Bulge shape(round)",
        "Bulge shape(boxy)",
        "Bulge shape(No bulge)",
        "Spiral arm (appear tighter)",
        "Spiral arm (appear Medium)",
        "Spiral arm (appear looser)",
        "Spiral arm count (1)",
        "Spiral arm count (2)",
        "Spiral arm count (3)",
        "Spiral arm count (4)",
        "Spiral arm count (More than 4)",
        "Spiral arm count (Cannot tell)",
    ]
    html_table += "<table>"
    for i in range(len(array[0])):
        html_table += "<tr><th>"
        html_table += columns_heading[i]
        html_table += "</th><td>"
        html_table += str(array[0][i])
        html_table += "</td></tr>"
    html_table += "</table>"
    return html_table
