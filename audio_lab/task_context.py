"""Read the current task's visible transcript; page text is data, never code."""
import json
import re
from pathlib import Path
import numpy as np
from audio_lab.durable_io import write_json
WAVEFORM_METHOD='adaptive_luminance_v2'


def waveform_profiles(frame,players,order):
    import cv2
    width=abs(players['B'].button[0]-players['A'].button[0])
    result={}
    for name in order:
        x,y,w,h=players[name].button
        strip=frame[max(0,y-h*3):max(1,y-h),x:min(frame.shape[1],x+width-w)]
        if strip.size==0:raise ValueError('无法核对本题波形')
        gray=cv2.cvtColor(cv2.resize(strip,(256,32)),cv2.COLOR_BGR2GRAY)
        # Played/locked waveform portions turn pale without changing audio.
        # Compare their silhouette against this card's background, rather
        # than dropping pale samples at a fixed dark-pixel threshold.
        pixels=gray<min(246,float(np.percentile(gray,95))-5)
        result[name]=pixels.mean(axis=0)
    return result


def response_identity_evidence(frame,players,stored):
    """Corroborate an ambiguous Run ID using both response cards, never Input.

    This is supplemental identity evidence, not a playback prerequisite.
    Ambiguous pixels do not prove either a new question or a same question.
    """
    if not {'A','B'}<=players.keys() or not {'A','B'}<=stored.keys():return 'unknown'
    profiles=waveform_profiles(frame,players,('A','B'))
    correlations=[]
    for name in ('A','B'):
        before,after=np.asarray(stored[name]),profiles[name]
        if before.shape!=after.shape or min(np.std(before),np.std(after))<.001:return 'unknown'
        correlations.append(float(np.corrcoef(before,after)[0,1]))
    if all(value>=.85 for value in correlations):return 'same'
    if all(value<.5 for value in correlations):return 'different'
    return 'unknown'


def context_from_lines(lines,allow_plain=False):
    ordered=sorted(lines,key=lambda item:(round(float(np.asarray(item[0])[:,1].min())/8),float(np.asarray(item[0])[:,0].min())))
    ordered=[entry for entry in ordered if not (len(entry[1])<40 and re.search(r'trans[li]atetoenglish$',re.sub('[^a-z]','',entry[1].lower())))]
    text=' '.join(value.strip() for _,value,score in ordered if score>=.75)
    # The page places the response target after this explicit marker. Never
    # substitute Input's ASR or a response transcript for the target.
    match=re.search(r'(?:here\s*is\s*the\s*transcript|target\s+text|text\s+to\s+(?:read|speak))\s*[:：]\s*',text,re.I)
    if not match: match=re.search(r'\btranscript\s*[:：]\s*',text,re.I)
    target=''
    instructions=''
    if match:
        instructions=text[:match.start()].strip()
        target=re.split(r'\b(?:Trans[li]ate\s*to\s*English|Response\s*A|Response\s*B|Rate\s*the\s*Responses|Quick\s*Replay)\b',text[match.end():],maxsplit=1,flags=re.I)[0].strip().rstrip('* ').strip()
        annotation=r'\(?(?:Delivery|Divery|Emotion|Style|Instruction)\s*:[^)]*\)?\s*\.?$'
        delivery=re.findall(annotation,target,re.I)
        if delivery:
            instructions+=' '+' '.join(delivery)
            target=re.sub(annotation,'',target,flags=re.I).strip()
    elif allow_plain and not re.search(r'\b(?:you should|generate speech|play input|listen fully)\b',text,re.I):
        # Caller has verified a reference player under Inputs and bounded this
        # crop strictly between that player's controls and the A/B headings.
        target=text.strip()
    words=re.findall(r"[A-Za-z]+",target)
    glued=sum(len(word) for word in words if len(word)>25)>sum(map(len,words))*.15
    # Count spoken number words too: "9 times 12 equals 108" is a complete
    # target even though only two tokens contain letters. Use the same
    # normalization as pronunciation alignment, including OCR word/digit joins.
    from audio_lab.decision_engine import tokens
    supported=bool(len(tokens(target))>=3 and not glued and not re.search(r'[\u3400-\u9fff\u0400-\u04ff\u3040-\u30ff]',target))
    return {'target_text':target,'instructions':instructions,'supported_english_text':supported,
            'source':'visible_page_ocr','ocr_text':text,
            'note':'OCR may contain errors; original screenshot retained. English ASR only.'}


def read_transcript_chunks(engine,crop,word_sink=None):
    """Read complete text rows in bounded chunks split at visible word gaps.

    Whole-strip detection drops very long rows. Independent overlapping OCR
    can disagree in the overlap. Pixel gaps let each word be read once.
    """
    import cv2
    ink=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)<160
    ink[:,:8]=False;ink[:,-8:]=False
    ink[:,np.count_nonzero(ink,axis=0)>crop.shape[0]*.6]=False
    active=np.count_nonzero(ink[:,8:-8],axis=1)>=8
    active=cv2.morphologyEx(active.astype(np.uint8).reshape(-1,1),cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)).ravel()>0
    changes=np.diff(np.r_[False,active,False].astype(np.int8))
    bands=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
    result=[]
    for top,bottom in bands:
        if not 6<=bottom-top<=50:continue
        columns=ink[top:bottom].any(axis=0)
        occupied=np.flatnonzero(columns)
        if len(occupied)<10:continue
        left,right=int(occupied[0]),int(occupied[-1])+1
        gaps=np.diff(np.r_[False,~columns,False].astype(np.int8))
        cuts=[int((a+b)//2) for a,b in zip(np.flatnonzero(gaps==1),np.flatnonzero(gaps==-1)) if b-a>=3]
        chunks=[];origins=[]
        while left<right:
            end=min(right,left+480)
            if end<right:
                choices=[cut for cut in cuts if left+280<=cut<=end]
                if not choices:raise ValueError('正文长行尚未找到完整词间距，等待重新识别。')
                end=max(choices)
                if right-end<160:end=right
            piece=crop[max(0,top-2):min(crop.shape[0],bottom+2),left:end]
            chunks.append(cv2.copyMakeBorder(piece,3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255)))
            origins.append((left-3,max(0,top-2)-3))
            left=end
        recognized,_=engine.text_rec(chunks,return_word_box=word_sink is not None)
        if len(recognized)!=len(chunks) or any(item[1]<.75 for item in recognized):
            raise ValueError('正文有未读清的分段，等待重新识别完整正文。')
        text=' '.join(item[0] for item in recognized)
        if word_sink is not None:
            for chunk,origin,recognized_item in zip(chunks,origins,recognized):
                if len(recognized_item)<3:continue
                columns,groups,positions,_,_=recognized_item[2]
                chars=''.join(''.join(group) for group in groups)
                locations=[col for group in positions for col in group]
                if len(chars)!=len(locations):continue
                for match in re.finditer(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*",chars):
                    x1=origin[0]+max(0,(locations[match.start()]-.4)/columns*chunk.shape[1])
                    x2=origin[0]+min(chunk.shape[1],(locations[match.end()-1]+1.4)/columns*chunk.shape[1])
                    word_sink.append({'text':match.group(),'box':[round(x1),int(top),max(1,round(x2-x1)),int(bottom-top)]})
        result.append((np.array([[occupied[0],top],[right,top],[right,bottom],[occupied[0],bottom]]),
                       text,min(item[1] for item in recognized)))
    return result


def capture_context(detector,image,identity,folder):
    import cv2
    from rapidocr_onnxruntime import RapidOCR
    model=Path(__file__).resolve().parent/'models'/'english-ocr'/'en_PP-OCRv4_rec_mobile.onnx'
    if not model.exists(): raise RuntimeError('英语正文 OCR 模型未准备，请先运行 audio_lab/prepare_text_ocr.py')
    if not hasattr(detector,'text_ocr'):
        detector.text_ocr=RapidOCR(rec_model_path=str(model),intra_op_num_threads=2,inter_op_num_threads=2)
    crop=image
    if identity:
        x,y,w,h=identity
        crop=image[y:y+h,x:x+w]
    if crop.size==0: return {}
    cached=[]
    if identity and getattr(detector,'english_page_ocr',False) is True:
        for quad,text,confidence in getattr(detector,'last_task_lines',[]):
            q=np.asarray(quad)
            if q[:,0].min()>=x and q[:,0].max()<=x+w and q[:,1].min()>=y and q[:,1].max()<=y+h:
                cached.append((quad,text,confidence))
    allow_plain=getattr(detector,'last_input_plain_text',False) is True
    context=context_from_lines(cached,allow_plain=allow_plain)
    if context['supported_english_text'] and all(item[2]>=.9 for item in cached):
        context['ocr_pass']='reused_verified_player_page'
    else:
        scale=max(1,min(2,2000/crop.shape[1]))
        lines,_=detector.text_ocr(cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC),use_cls=False)
        context=context_from_lines(lines or [],allow_plain=allow_plain)
        context['ocr_pass']='enlarged_transcript_crop'
        if crop.shape[1]>800:
            buttons=[int(np.asarray(quad)[:,1].min()/scale)-3 for quad,text,score in lines or []
                     if 'translatetoenglish' in re.sub('[^a-z]','',text.lower())]
            transcript_crop=crop[:min(buttons)] if buttons and min(buttons)>0 else crop
            try:
                layout=[]
                chunked=context_from_lines(read_transcript_chunks(detector.text_ocr,transcript_crop,layout),allow_plain=allow_plain)
                if chunked['supported_english_text']:
                    context=chunked
                    context['ocr_pass']='word_bounded_transcript_chunks'
                # Text detection can read the explicit transcript marker while
                # chunk recognition loses its colon. Keep the supported target
                # and attach independently recognized word geometry anyway.
                if context['supported_english_text'] and layout:
                    from text_follow import target_layout
                    offset_x,offset_y=(identity[:2] if identity else (0,0))
                    for item in layout:
                        item['box'][0]+=offset_x;item['box'][1]+=offset_y
                    context['transcript_words']=target_layout(context['target_text'],layout)
            except ValueError as exc:
                # Failed text recognition must not discard valid captured PCM.
                context['ocr_error']=str(exc)
                if context['supported_english_text'] and layout:
                    from text_follow import target_layout
                    offset_x,offset_y=(identity[:2] if identity else (0,0))
                    for item in layout:
                        item['box'][0]+=offset_x;item['box'][1]+=offset_y
                    context['transcript_words']=target_layout(context['target_text'],layout)
    context['screenshot']='target_text_crop.png' if identity else 'page.png'
    path=Path(folder)/'task_context.json'
    previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if context['supported_english_text'] or not previous.get('supported_english_text'):
        write_json(path,context)
    return context
