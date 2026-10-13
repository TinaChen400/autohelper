"""Auditable automatic candidate ratings from this task's local evidence.

These are uncalibrated predictions, not verified listening judgments. Missing
recordings, missing target text, and failed models never turn into fixed votes.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import wave
import numpy as np

VERSION='local-candidates-v4'
NAMES={'quality':'音质','speaker':'音色相似度','pronunciation':'发音忠实度','naturalness':'自然度'}


def item(choice,reason,evidence,uncertainty='uncalibrated'):
    return {'judgment':choice,'reason':reason,'evidence':evidence,
            'uncertainty':uncertainty,'method':VERSION,'automatic_candidate':True}


def audio_metrics(path):
    with wave.open(str(path)) as wav:
        if wav.getsampwidth()!=2: raise ValueError('录音不是 PCM16')
        rate=wav.getframerate()
        samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').reshape(-1,wav.getnchannels()).astype(np.float32)/32768
    if len(samples)<rate*.3 or not np.isfinite(samples).all(): raise ValueError('录音为空或过短')
    audio=samples.mean(axis=1)
    width=max(1,round(rate*.02))
    frames=audio[:len(audio)//width*width].reshape(-1,width)
    rms=np.sqrt(np.mean(frames**2,axis=1))
    active=np.flatnonzero(rms>max(.003,float(rms.max())*.025))
    if len(active)<3: raise ValueError('录音中没有足够语音信号')
    peak=np.max(np.abs(samples),axis=1)
    clipped=peak>.995
    # Consecutive saturated samples, not an isolated transient peak.
    runs=np.convolve(clipped.astype(np.int8),np.ones(3,dtype=np.int8),mode='valid')>=3
    zeros=np.max(np.abs(samples),axis=1)<1/32768
    lo=int(active[0]*width);hi=int((active[-1]+1)*width)
    changes=np.diff(np.r_[False,zeros[lo:hi],False].astype(np.int8))
    gaps=(np.flatnonzero(changes==-1)-np.flatnonzero(changes==1))/rate
    return {'seconds':len(audio)/rate,'rms':float(np.sqrt(np.mean(audio**2))),
            'clipped_fraction':float(np.mean(clipped)),'clipped_run_fraction':float(np.mean(runs)),
            'internal_digital_zero_gaps_seconds':[round(float(gap),3) for gap in gaps if gap>=.12],
            'sha256':sha256(samples.tobytes()).hexdigest()}


def quality_candidate(scores,metrics):
    severity={};flags={}
    for name in ('A','B'):
        q=scores[name];m=metrics[name]
        for field in ('MOS_OVRL','MOS_SIG','MOS_DISC','MOS_NOISE'):
            if not np.isfinite(q[field]): raise ValueError('音质分数无效')
        # Severe low scores support a predicted defect; small MOS differences
        # alone do not create a preference. All thresholds are policy values.
        problems=[]
        if m['clipped_run_fraction']>.001: problems.append(('连续数字削波',min(2,m['clipped_run_fraction']*100)))
        for field,threshold,label in [('MOS_SIG',2.4,'信号质量低'),('MOS_DISC',2.4,'失真预测偏高'),('MOS_NOISE',2.6,'噪声预测偏高')]:
            if q[field]<threshold: problems.append((label,threshold-q[field]+.35))
        if q['MOS_OVRL']<2.3: problems.append(('整体音质预测低',2.3-q['MOS_OVRL']+.35))
        if any(gap>.3 for gap in m['internal_digital_zero_gaps_seconds']) and problems:
            problems.append(('内部数字静音，可能断流或有意停顿',.3))
        flags[name]=problems
        severity[name]=sum(value for _,value in problems)
    a,b=severity['A'],severity['B']
    if max(a,b)==0: choice='Both Good'
    elif min(a,b)==0 or abs(a-b)>=.4: choice='A' if a<b else 'B'
    else: choice='Both Bad'
    evidence={'scores':scores,'audio_metrics':metrics,'predicted_defects':flags,'severity':severity}
    reason=f'自动候选 {choice}；缺陷代理严重度 A={a:.2f}、B={b:.2f}；综合 SigMOS 子项与削波/数字静音检测，尚未人工听感校准'
    return item(choice,reason,evidence),severity


def number_words(value):
    small='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    tens='zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
    if value<20:return small[value]
    if value<100:return tens[value//10]+(' '+small[value%10] if value%10 else '')
    if value<1000:return small[value//100]+' hundred'+(' '+number_words(value%100) if value%100 else '')
    if value<1000000:return number_words(value//1000)+' thousand'+(' '+number_words(value%1000) if value%1000 else '')
    return ' '.join(small[int(c)] for c in str(value))


def tokens(text):
    # Preserve signs and expand written units before stripping punctuation.
    # OCR/ASR often differ only in these spoken representations.
    text=re.sub(r'(?i)(?<![a-z])ghz(?![a-z])','gigahertz',text)
    text=re.sub(r'(?i)(?<![a-z])dbm(?![a-z])','dbm',text)
    text=re.sub(r'(?<=[a-z])(?=[A-Z][a-z])',' ',text)
    text=text.lower().replace('’',"'")
    text=re.sub(r'(?<!\w)[−-](?=\d)','minus ',text)
    text=re.sub(r'(?<!\w)(\d+)\.(\d+)',lambda m:number_words(int(m[1]))+' point '+
                ' '.join(number_words(int(digit)) for digit in m[2])+' ',text)
    text=re.sub(r'\bgiga\s+hertz\b','gigahertz',text)
    # Numeric OCR confusion is restricted to mixed digit/O runs, not names.
    text=re.sub(r'\b(?=[0-9o]*\d)[0-9o]+\b',lambda m:m.group().replace('o','0'),text)
    for joined,expanded in {"theyre":"they are","youre":"you are","isnt":"is not","doesnt":"does not","dont":"do not"}.items():
        text=re.sub(r'\b'+joined+r'\b',expanded,text)
    text=re.sub(r'\bits(?=\s+(?:just|not|been|going|really)\b)','it is',text)
    # Letter/digit boundaries are formatting, not pronunciation differences:
    # CYP2D6 and CYP 2D6 must have the same word representation.
    text=re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])',' ',text)
    contractions={"it's":"it is","he's":"he is","she's":"she is","that's":"that is","there's":"there is",
        "i'm":"i am","you're":"you are","we're":"we are","they're":"they are","don't":"do not",
        "doesn't":"does not","isn't":"is not","can't":"can not","won't":"will not","i've":"i have"}
    for word,replacement in contractions.items(): text=re.sub(r'\b'+re.escape(word)+r'\b',replacement,text)
    text=re.sub(r'\d+',lambda match:number_words(int(match.group())),text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?",text)


def align(reference,hypothesis):
    rows=[[None]*(len(hypothesis)+1) for _ in range(len(reference)+1)]
    rows[0][0]=(0,[])
    for i in range(len(reference)+1):
        for j in range(len(hypothesis)+1):
            if i==j==0:continue
            choices=[]
            if i: choices.append((rows[i-1][j][0]+1,rows[i-1][j][1]+[('delete',reference[i-1],'')]))
            if j: choices.append((rows[i][j-1][0]+1,rows[i][j-1][1]+[('insert','',hypothesis[j-1])]))
            if i and j:
                equal=reference[i-1]==hypothesis[j-1]
                choices.append((rows[i-1][j-1][0]+(not equal),rows[i-1][j-1][1]+([] if equal else [('replace',reference[i-1],hypothesis[j-1])])))
            rows[i][j]=min(choices,key=lambda candidate:candidate[0])
    count,edits=rows[-1][-1]
    return {'word_error_rate':count/max(1,len(reference)),'edits':edits,'reference_word_count':len(reference)}


def pronunciation_candidate(target,transcripts):
    reference=tokens(target)
    if len(reference)<3: raise ValueError('页面目标正文未完整识别，不用 Input 或 A/B 转写代替正文')
    details={name:align(reference,tokens(transcripts[name])) for name in ('A','B')}
    a,b=(details[name]['word_error_rate'] for name in ('A','B'))
    common=list((Counter(tuple(edit) for edit in details['A']['edits']) & Counter(tuple(edit) for edit in details['B']['edits'])).elements())
    shared=len(common)/len(reference)
    if max(a,b)<=.15: choice='Both Good'
    elif abs(a-b)>=.08: choice='A' if a<b else 'B'
    elif min(a,b)>=.3: choice='Both Bad'
    else: choice='A' if a<b else 'B' if b<a else 'Both Good'
    candidate=choice
    # ASR spelling/splitting errors, especially shared proper names, cannot
    # establish that both recordings mispronounced the text. Keep the raw
    # alignment but abstain rather than convert uncertainty into a bad vote.
    needs_review=choice=='Both Bad' or shared>.1 and min(a,b)>.15
    if needs_review:choice='需复核'
    warning='；两段存在共同差异，可能是 OCR/ASR 或读法差异' if shared>.1 else ''
    if needs_review:warning+='；仅凭转写差异不足以判定发音错误，须人工回听确认'
    return item(choice,f'自动候选 {choice}；与当题正文的归一化词差异 A={a:.1%}、B={b:.1%}{warning}；ASR 内容一致性代理，未独立验证音素、重音与专名读法',
        {'target_text':target,'transcripts':transcripts,'alignment':details,'shared_edits':common,
         'asr_only_candidate':candidate,'requires_listening_review':needs_review},
        'high_uncertainty' if needs_review or shared>.1 or abs(a-b)<.08 and max(a,b)>.15 else 'uncalibrated'),details


def reviewed_pronunciation(target,primary,secondary):
    """Resolve a comparative preference only when both decodes support it.

    Shared OCR/name/code-reading disagreements cannot establish Both Bad.
    Keep them in the audit, and compare the remaining response-specific edits.
    """
    original,details=pronunciation_candidate(target,primary)
    if original['judgment']!='需复核':return original,details
    other,other_details=pronunciation_candidate(target,secondary)
    if other['judgment']=='Both Good':
        other['reason']+='；独立解码复核已覆盖 A/B，保留原转写差异供检查'
        other['evidence']['primary_review']=original
        other['evidence']['review_method']='independent_decode'
        return other,other_details
    evidence=[];winners=[]
    for transcripts in (primary,secondary):
        _,aligned=pronunciation_candidate(target,transcripts)
        a,b=(Counter(tuple(edit) for edit in aligned[name]['edits']) for name in ('A','B'))
        common=a&b;unique={name:sum((counts-common).values()) for name,counts in (('A',a),('B',b))}
        winners.append('A' if unique['A']<unique['B'] else 'B' if unique['B']<unique['A'] else None)
        evidence.append({'unique_edits':unique,'shared_edits':list(common.elements()),'alignment':aligned})
    if winners[0] and winners[0]==winners[1]:
        winner=winners[0]
        return item(winner,f'自动候选 {winner}；两次独立解码均显示该段独有的内容差异更少；共同的 OCR/专名/代码读法差异未用于判坏，音素发音仍有不确定性',
            {'review_method':'two_decode_relative_alignment','passes':evidence,'transcripts':[primary,secondary]},'high_uncertainty'),other_details
    original['evidence']['review_attempted']=True
    original['evidence']['secondary_transcripts']=secondary
    original['evidence']['review_result']='no_supported_preference'
    return original,details


def tie_winner(metrics):
    # Exact evidence ties have no supported winner. The page forces A/B;
    # content hashes give reproducible tie-breaking without always favoring A.
    return 'A' if metrics['A']['sha256']<=metrics['B']['sha256'] else 'B'


def build_decisions(folder):
    folder=Path(folder)
    (folder/'evaluation_decisions.json').unlink(missing_ok=True)
    context=json.loads((folder/'task_context.json').read_text(encoding='utf-8'))
    if not context.get('supported_english_text'): raise ValueError('缺少可用英语目标正文，不能完成发音及整体评价')
    report=json.loads((folder/'model_comparison.json').read_text(encoding='utf-8'))
    natural=json.loads((folder/'naturalness.json').read_text(encoding='utf-8'))['scores']
    session=json.loads((folder/'session.json').read_text(encoding='utf-8'))
    metrics={name:audio_metrics(folder/f'{name}.wav') for name in ('A','B')}
    judgments={}
    judgments['quality'],severity=quality_candidate({name:report['quality'][name] for name in ('A','B')},metrics)
    judgments['pronunciation'],alignment=pronunciation_candidate(context['target_text'],{name:report['transcripts'][name] for name in ('A','B')})
    review_path=folder/'pronunciation_review.json'
    if judgments['pronunciation']['judgment']=='需复核' and review_path.exists():
        review=json.loads(review_path.read_text(encoding='utf-8'))
        if review.get('state')=='complete' and all(review.get('transcripts',{}).get(name) for name in ('A','B')):
            judgments['pronunciation'],alignment=reviewed_pronunciation(context['target_text'],
                {name:report['transcripts'][name] for name in ('A','B')},review['transcripts'])
    n={name:float(natural[name]['mos']) for name in ('A','B')}
    if not all(np.isfinite(value) for value in n.values()): raise ValueError('自然度分数无效')
    nw='A' if n['A']>n['B'] else 'B' if n['B']>n['A'] else tie_winner(metrics)
    judgments['naturalness']=item(nw,f'自动候选 {nw}；UTMOSv2 A={n["A"]:.3f}、B={n["B"]:.3f}；分差小意味着倾向弱，未验证语义重音',n,
        'high_uncertainty' if abs(n['A']-n['B'])<.15 else 'uncalibrated')
    contrasts=[('naturalness',nw,abs(n['A']-n['B'])/.15)]
    if 'Input' in session['player_order']:
        s=report['speaker_similarity']
        if not all(np.isfinite(s[name]) for name in ('A','B')): raise ValueError('音色分数无效')
        sw='A' if s['A']>s['B'] else 'B' if s['B']>s['A'] else ('Both Good' if s['A']>=.65 else 'Both Bad')
        judgments['speaker']=item(sw,f'自动候选 {sw}；ECAPA A={s["A"]:.3f}、B={s["B"]:.3f}；与 Input 比较，未校准',s)
        if sw in ('A','B'): contrasts.append(('speaker',sw,abs(s['A']-s['B'])/.04))
    else: judgments['speaker']=item('不适用','本题无参考音频',{})
    # Compare the strongest supported contrast, rather than average raw MOS,
    # cosine similarity and word errors. Severe defects take precedence.
    difference=alignment['B']['word_error_rate']-alignment['A']['word_error_rate']
    if judgments['quality']['judgment'] in ('A','B'):
        contrasts.append(('quality',judgments['quality']['judgment'],abs(severity['A']-severity['B'])/.4))
    if judgments['pronunciation']['judgment'] in ('A','B'):
        contrasts.append(('pronunciation',judgments['pronunciation']['judgment'],abs(difference)/.08))
    if max(severity.values())>=1 and abs(severity['A']-severity['B'])>=.4:
        dominant='quality';winner='A' if severity['A']<severity['B'] else 'B';strength=abs(severity['A']-severity['B'])/.4
    elif judgments['pronunciation']['judgment'] in ('A','B') and max(alignment[name]['word_error_rate'] for name in ('A','B'))>=.3 and abs(difference)>=.15:
        dominant='pronunciation';winner='A' if difference>0 else 'B';strength=abs(difference)/.08
    else: dominant,winner,strength=max(contrasts,key=lambda contrast:contrast[2])
    if strength==0: winner=tie_winner(metrics)
    reason=f'自动候选 {winner}；主导因素：{NAMES[dominant]}；'+('全部可用证据等强，按音频内容哈希作可复现的强制二选一' if strength==0 else '优先避开明显预测缺陷；其余按最强标准化差异决定，不平均原始模型分数')
    judgments['overall']=item(winner,reason,{'contrasts':contrasts,'dominant_strength':strength},'high_uncertainty' if strength<1 else 'uncalibrated')
    judgments['overall']['dominant_factor']=dominant
    if re.search(r'\b(?:delivery|emotion|style)\s*:',context.get('instructions',''),re.I):
        judgments['naturalness']['reason']+='；页面附带风格要求，当前未单独验证情绪遵循度'
        judgments['naturalness']['uncertainty']='high_uncertainty'
    result={'version':VERSION,'state':'complete_candidates','judgments':judgments,'task_context':context,
        'limitations':['自动候选尚未按人工标签校准，不保证判断正确。','发音项以正文/ASR对齐代理，不能证明音素发音正确。',
            '音质由模型与数字缺陷指标预测，传输伪影与有意停顿仍可能混淆。'],
        'policy':{'quality_subscore_thresholds':{'MOS_SIG':2.4,'MOS_DISC':2.4,'MOS_NOISE':2.6,'MOS_OVRL':2.3},
            'contrast_scales':{'speaker':.04,'naturalness':.15,'pronunciation':.08,'quality':.4},
            'note':'工程策略阈值，不是经人工数据验证的置信度。'}}
    (folder/'evaluation_decisions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    print(json.dumps(build_decisions(parser.parse_args().folder),ensure_ascii=True))
