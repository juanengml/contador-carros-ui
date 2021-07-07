import gradio as gr
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

def classify_image(opt,inp):
  bbox, label, conf = cv.detect_common_objects(inp)
  output_image = draw_bbox(inp, bbox, label, conf)
  return output_image,label.count(opt)

lbl = [ 
         gr.inputs.Dropdown(label="Selecione o label para contagem ",choices=["car", "person", "bus", "train"]),
         gr.inputs.Image(label="Upload image",shape=(720, 420))
       ]

outputs = [  gr.outputs.Image(label="Output"), 
             gr.outputs.Label(type="auto", label="Total")
          ]


gr.Interface(fn=classify_image, inputs=lbl, outputs=outputs, capture_session=True, layout="vertical").launch()
