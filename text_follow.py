"""Map recognized speech phrases to OCR word boxes, never elapsed-time guesses."""
from difflib import SequenceMatcher
import re


def words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*",text)


def normalized(text):
    return re.sub('[^a-z0-9]','',text.lower())


def target_layout(target,observed):
    wanted=words(target)
    matcher=SequenceMatcher(None,list(map(normalized,wanted)),[normalized(item['text']) for item in observed],autojunk=False)
    result=[]
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            result.append({**observed[block.b+offset],'target_index':block.a+offset})
    return sorted(result,key=lambda item:item['target_index'])


def phrase_position(target,spoken,previous=0):
    intended=list(map(normalized,words(target)));heard=list(map(normalized,words(spoken)))
    if len(heard)<2:return None
    matches=SequenceMatcher(None,intended,heard,autojunk=False).get_matching_blocks()
    candidates=[b for b in matches if b.size>=2 and b.b+b.size>=len(heard)-2 and b.a+b.size-1>=previous-2]
    if not candidates:return None
    match=max(candidates,key=lambda b:(b.b+b.size,b.size,-abs(b.a-previous)))
    return match.a+match.size-1
