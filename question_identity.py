"""Three-way task identity; uncertain OCR never starts another full recording."""


def one_digit_edit(first,second):
    if first==second:return True
    if abs(len(first)-len(second))>1:return False
    if len(first)==len(second):return sum(a!=b for a,b in zip(first,second))<=1
    shorter,longer=sorted((first,second),key=len)
    return any(longer[:i]+longer[i+1:]==shorter for i in range(len(longer)))


def relation(expected,observed,content='unknown'):
    before,after=dict(expected or ()),dict(observed or ())
    shared=before.keys()&after.keys()
    mismatched={key for key in shared if before[key]!=after[key]}
    if content=='same':return 'same'
    if content=='different' and mismatched:return 'different'
    if not before or not after:return 'unknown'
    if not mismatched:
        if before.get('prompt') and before.get('prompt')==after.get('prompt'):return 'same'
        return 'same' if before.keys()<=after.keys() else 'unknown'
    if 'prompt' in mismatched:
        if one_digit_edit(before['prompt'],after['prompt']):return 'unknown'
        return 'different'
    # Runs are supplemental. A changed Run alone needs response evidence.
    return 'unknown'
