"""Read rating rows and their selected states from visible pixels."""
import re
import cv2
import numpy as np


def dimension_key(text):
    name=re.sub(r'[^a-z]','',text.lower())
    return {'audioandrecordingquality':'quality','audioquality':'quality',
            'personalikenessspeakersimilarity':'speaker','speakersimilarity':'speaker',
            'pronunciationfaithfulness':'pronunciation','naturalness':'naturalness',
            'overallpreference':'overall','dominantfactor':'factor'}.get(name)


def judgment_name(option):
    return {'Response A':'A','Response B':'B','N/A':'不适用'}.get(option,dimension_key(option) or option)


def question_marker(lines):
    text=' '.join(text for _,text,score in lines if score>=.75)
    return tuple((name.lower(),number) for name,number in re.findall(r'(Prompt|Run)\s*#?\s*(\d+)',text,re.I))


def marker_changed(expected, observed, *, form_reset=False):
    """A Run OCR variation alone cannot send an evaluated task to recording.

    Runs can differ under the same prompt, but that boundary needs the form
    reset after submission as corroboration. Missing IDs are inconclusive.
    """
    expected,observed=dict(expected or ()),dict(observed or ())
    if 'prompt' in expected and 'prompt' in observed:
        if expected['prompt']!=observed['prompt']:
            from question_identity import one_digit_edit
            truncated=(min(len(expected['prompt']),len(observed['prompt']))>=5 and
                       len(expected['prompt'])!=len(observed['prompt']) and
                       one_digit_edit(expected['prompt'],observed['prompt']))
            return bool(form_reset or not truncated)
    return bool(form_reset and 'run' in expected and 'run' in observed
                and expected['run']!=observed['run'])


def marker_matches(expected, observed):
    expected,observed=dict(expected or ()),dict(observed or ())
    if 'prompt' in expected:
        return observed.get('prompt')==expected['prompt']
    return bool(expected and expected.items()<=observed.items())


def button_region(image, text_box):
    """Find the enclosing button border; coordinates never come from a fixed page."""
    x,y,w,h=map(int,text_box)
    pad=max(8,round(h*1.3))
    left,top=max(0,x-pad),max(0,y-pad)
    right,bottom=min(image.shape[1],x+w+pad),min(image.shape[0],y+h+pad)
    crop=image[top:bottom,left:right]
    edges=cv2.Canny(crop,8,30)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    filled=(((hsv[:,:,1]>70)&(hsv[:,:,2]>100))|(hsv[:,:,2]<80)|
            ((hsv[:,:,1]<35)&(hsv[:,:,2]>=145)&(hsv[:,:,2]<240))).astype(np.uint8)*255
    filled=cv2.morphologyEx(filled,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    fill_contours,_=cv2.findContours(filled,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=list(contours)+list(fill_contours)
    candidates=[]
    for contour in contours:
        bx,by,bw,bh=cv2.boundingRect(contour)
        # A contour cut by the search crop can include the neighboring selected
        # button. It is not a complete enclosing border for this option.
        if bx<=0 or by<=0 or bx+bw>=crop.shape[1] or by+bh>=crop.shape[0]: continue
        bx+=left;by+=top
        if bx<=x+1 and by<=y and bx+bw>=x+w-1 and by+bh>=y+h and bw>=w+3 and bh>=h+4:
            candidates.append((bx,by,bw,bh))
    return min(candidates,key=lambda b:b[2]*b[3]) if candidates else None


def selected_state(image, options):
    evidence={}
    for name,text_box in options.items():
        border=button_region(image,text_box)
        if border is None:
            tx,ty,tw,th=map(int,text_box)
            pad=max(5,round(th*.4))
            if tx-pad<0 or ty-pad<0 or tx+tw+pad>=image.shape[1] or ty+th+pad>=image.shape[0]:
                evidence[name]={'state':'unknown'};continue
            border=(tx-pad,ty-pad,tw+pad*2,th+pad*2)
        x,y,w,h=border
        inset=max(2,round(h*.1))
        crop=image[y+inset:y+h-inset,x+inset:x+w-inset]
        mask=np.ones(crop.shape[:2],np.uint8)
        tx,ty,tw,th=text_box
        a,b=max(0,tx-x-inset-1),max(0,ty-y-inset-1)
        mask[b:min(crop.shape[0],ty+th-y-inset+1),a:min(crop.shape[1],tx+tw-x-inset+1)]=0
        pixels=crop[mask.astype(bool)]
        if len(pixels)<8:
            whole=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            broad_color=float(np.mean(whole[:,:,1]>70))
            broad_dark=float(np.mean(whole[:,:,2]<145))
            broad_gray=float(np.mean((whole[:,:,1]<35)&(whole[:,:,2]>=145)&(whole[:,:,2]<243)))
            white=float(np.mean((whole[:,:,1]<8)&(whole[:,:,2]>248)))
            filled=broad_color>.65 or broad_dark>.75 or broad_gray>.75 and white<.1
            evidence[name]={'state':'selected' if filled else 'unknown','button_box':border,
                'surface_fill_fraction':broad_color,'surface_dark_fraction':broad_dark}
            continue
        hsv=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3)
        fill=float(np.mean(hsv[:,1]>70))
        dark=float(np.mean(hsv[:,2]<145))
        tint=float(np.mean(hsv[:,1]>12))
        neutral=float(np.mean((hsv[:,1]<35)&(hsv[:,2]>=145)&(hsv[:,2]<240)))
        whole=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        white=float(np.mean((whole[:,:,1]<8)&(whole[:,:,2]>248)))
        neutral_surface=float(np.mean((whole[:,:,1]<35)&(whole[:,:,2]>=145)&(whole[:,:,2]<243)))
        # A nearly button-wide OCR box can mask all interior background and
        # leave only antialiased border pixels. In that case require a broad
        # neutral fill across the actual enclosed surface and almost no white.
        surface_gray=(neutral_surface>.75 and white<.1
                      and np.count_nonzero(mask)<crop.shape[0]*crop.shape[1]*.25)
        # Colored text and outlines alone are insufficient: require broad fill
        # outside the OCR text box, or a dark filled button.
        state='selected' if fill>.65 or dark>.75 or neutral>.7 or surface_gray else 'clear' if white>.55 or fill<.12 and dark<.12 and tint<.2 and neutral<.2 else 'unknown'
        evidence[name]={'state':state,'button_box':border,'fill_fraction':fill,'dark_fraction':dark,'tint_fraction':tint,'neutral_fraction':neutral,'neutral_surface_fraction':neutral_surface}
    selected=[name for name,item in evidence.items() if item['state']=='selected']
    unknown=any(item['state']=='unknown' for item in evidence.values())
    if len(selected)==1 and not unknown: return selected[0],'selected',evidence
    if not selected and not unknown: return None,'none',evidence
    return None,'ambiguous',evidence


def submit_enabled(image,boxes):
    for x,y,w,h in boxes:
        pad=max(4,round(h*.3))
        if x-pad<0 or y-pad<0 or x+w+pad>image.shape[1] or y+h+pad>image.shape[0]: continue
        crop=image[y-pad:y+h+pad,x-pad:x+w+pad]
        mask=np.ones(crop.shape[:2],bool);mask[pad:pad+h,pad:pad+w]=False
        pixels=crop[mask]
        hsv=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3)
        if float(np.mean((hsv[:,2]<70)|((hsv[:,1]>45)&(hsv[:,2]<235))))>.7: return True
    return False


def option_name(text):
    normalized = re.sub(r'[^a-z]', '', text.lower())
    return {'responsea': 'Response A', 'responseb': 'Response B', 'bothgood': 'Both Good',
            'bothbad': 'Both Bad', 'na': 'N/A'}.get(normalized)


def dimensions_from_lines(lines, shape):
    items = []
    for quad, text, confidence in lines:
        if confidence < .75: continue
        q = np.asarray(quad)
        x, y = q.min(axis=0); right, bottom = q.max(axis=0)
        items.append({'text': text.strip(), 'box': (int(x), int(y), int(right-x), int(bottom-y)), 'option': option_name(text)})
    options = sorted([row for row in items if row['option']], key=lambda row: row['box'][1])
    groups = []
    for row in options:
        cy = row['box'][1]+row['box'][3]/2
        for group in groups:
            if abs(cy-group[0]['box'][1]-group[0]['box'][3]/2) <= max(6, row['box'][3]):
                group.append(row); break
        else: groups.append([row])
    dimensions = []
    for group in groups:
        names = {row['option'] for row in group}
        if len(names) < 2 or not {'Response A','Response B'}.issubset(names): continue
        left = min(row['box'][0] for row in group)
        cy = sum(row['box'][1]+row['box'][3]/2 for row in group)/len(group)
        labels = [row for row in items if not row['option'] and row['box'][0]+row['box'][2] < left
                  and abs(row['box'][1]+row['box'][3]/2-cy) < max(8, row['box'][3]*1.3)
                  and 'rate the' not in row['text'].lower()]
        labels=[row for row in labels if dimension_key(row['text'].rstrip(' *')) in ('quality','speaker','pronunciation','naturalness','overall')]
        if not labels: continue
        label = min(labels, key=lambda row: abs(row['box'][1]+row['box'][3]/2-cy))
        descriptions = [row['text'] for row in items if not row['option']
                        and row['box'][0] < left and row['box'][1] > label['box'][1]+label['box'][3]
                        and abs(row['box'][0]-label['box'][0]) < max(20, label['box'][3]*4)
                        and row['box'][1] < label['box'][1]+label['box'][3]*3]
        dimensions.append({'name': label['text'].rstrip(' *'), 'key':dimension_key(label['text']), 'description': ' '.join(descriptions),
                           'label_box': label['box'], 'options': {row['option']: row['box'] for row in group}})
    for label in items:
        if dimension_key(label['text'])!='factor': continue
        x,y,w,h=label['box']
        choices=[item for item in items if dimension_key(item['text']) in ('quality','speaker','pronunciation','naturalness')
                 and y+h<item['box'][1]<y+h*5 and item['box'][0]>=x-5]
        if len({dimension_key(item['text']) for item in choices})>=2:
            dimensions.append({'name':label['text'].rstrip(' *'),'key':'factor','description':'',
                'label_box':label['box'],'options':{item['text']:item['box'] for item in choices}})
    return dimensions


def enclosed_form_boxes(image):
    height,width=image.shape[:2]
    contours,_=cv2.findContours(cv2.Canny(image,8,30),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    forms=[cv2.boundingRect(c) for c in contours]
    return [(x,y,w,h) for x,y,w,h in forms if width*.6<w<width*.95 and h>height*.25
           and y>height*.15 and x>0 and x+w<width-1 and y+h<height-1]


def selected_button_lines(detector,image):
    """Read white glyphs directly from actual filled-button contours."""
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    fill=(((hsv[:,:,1]>90)&(hsv[:,:,2]>80))|(hsv[:,:,2]<75)).astype(np.uint8)*255
    contours,_=cv2.findContours(fill,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lines=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not (45<=w<=image.shape[1]*.32 and 16<=h<=65 and y>image.shape[0]*.15):continue
        if cv2.contourArea(contour)<w*h*.65:continue
        crop=image[y:y+h,x:x+w]
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        text=np.clip((255-gray.astype(float))*255/max(1,255-float(np.median(gray))),0,255).astype(np.uint8)
        text[:2]=255;text[-2:]=255;text[:,:2]=255;text[:,-2:]=255
        ink=np.argwhere(text<128)
        if len(ink)<8:continue
        padded=cv2.copyMakeBorder(text,3,3,3,3,cv2.BORDER_CONSTANT,value=255)
        result,_=detector.ocr.text_rec([cv2.cvtColor(padded,cv2.COLOR_GRAY2BGR)])
        if not result:continue
        value,score=result[0][:2]
        normalized=re.sub('[^a-z]','',value.lower())
        if score<.8 or normalized not in ('responsea','responseb','na','naturalness','audioandrecordingquality',
            'personalikenessspeakersimilarity','pronunciationfaithfulness'):continue
        top,left=ink.min(axis=0);bottom,right=ink.max(axis=0)
        quad=np.array([[x+left,y+top],[x+right+1,y+top],[x+right+1,y+bottom+1],[x+left,y+bottom+1]])
        lines.append((quad,value,float(score)))
    return lines


def read_rating_screen(detector, image):
    # A visible, enclosed wide form supplies a screenshot-derived crop. Never
    # infer option positions from it; row labels/options still require OCR.
    forms=enclosed_form_boxes(image)
    top=min(forms,key=lambda b:b[2]*b[3])[1]-4 if forms else 0
    crop=image[max(0,top):]
    scale = min(1.5, 1600/image.shape[1])
    scaled=cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if getattr(detector,'fast_ui_ocr',False) is True:
        lines=detector.rating_lines(scaled)
    else:
        lines,_=detector.ocr(scaled,use_cls=False)
    lines = [(np.asarray(q)/scale+np.array([0,max(0,top)]), text, score) for q, text, score in (lines or [])]
    if getattr(detector,'fast_ui_ocr',False) is True:
        for quad,text,score in selected_button_lines(detector,image):
            center=quad.mean(axis=0)
            if not any(re.sub('[^a-z]','',old.lower())==re.sub('[^a-z]','',text.lower()) and
                       np.linalg.norm(np.asarray(q).mean(axis=0)-center)<20 for q,old,_ in lines):
                lines.append((quad,text,score))
    dimensions = dimensions_from_lines(lines, image.shape)
    submit_boxes=[];guard_boxes=[]
    for q,text,score in lines:
        if question_marker([(q,text,score)]):
            x,y=np.asarray(q).min(axis=0);right,bottom=np.asarray(q).max(axis=0)
            guard_boxes.append((int(x),int(y),int(right-x),int(bottom-y)))
        if re.sub(r'[^a-z]','',text.lower()) in ('submit','submitvote','submitrating','submitratings') and score>=.85:
            q=np.asarray(q);x,y=q.min(axis=0);right,bottom=q.max(axis=0)
            submit_boxes.append((int(x),int(y),int(right-x),int(bottom-y)))
    for row in dimensions:
        selected,state,evidence=selected_state(image,row['options'])
        row.update(selected=selected,selection_state=state,selection_evidence=evidence)
    return {'rows':dimensions,'submit_boxes':submit_boxes,'submit_enabled':submit_enabled(image,submit_boxes),
            'question_marker':question_marker(lines),'guard_boxes':guard_boxes}


def refresh_selections(image,rows):
    refreshed=[]
    for row in rows:
        selected,state,evidence=selected_state(image,row['options'])
        refreshed.append({**row,'selected':selected,'selection_state':state,'selection_evidence':evidence})
    return refreshed


def read_rating_page(detector,image):
    screen=read_rating_screen(detector,image)
    return screen['rows'],bool(screen['submit_boxes'])
