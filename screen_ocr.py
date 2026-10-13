"""Recognize short UI labels without decoding long paragraphs and waveforms."""
import cv2
import numpy as np


def ui_text_lines(engine,image,max_aspect=42):
    # Detection is inexpensive. Long transcript/description lines dominate
    # recognition and pad every other item in their recognition batch.
    boxes,_=engine(image,use_rec=False,use_cls=False)
    kept=[];crops=[]
    for box in boxes or []:
        q=np.asarray(box,dtype=np.float32)
        x,y=np.floor(q.min(axis=0)).astype(int)
        right,bottom=np.ceil(q.max(axis=0)).astype(int)
        x,y=max(0,x),max(0,y)
        right,bottom=min(image.shape[1],right),min(image.shape[0],bottom)
        w,h=right-x,bottom-y
        if h<3 or w<3 or w/h>max_aspect:continue
        # Retain the detector's quadrilateral and use its perspective crop,
        # exactly as the normal RapidOCR recognition path does.
        kept.append(q)
    if not kept:return []
    crops=engine.get_crop_img_list(image,np.asarray(kept))
    recognized,_=engine.text_rec(crops)
    return [(q,text,float(score)) for q,(text,score) in zip(kept,recognized) if score>=.5]
