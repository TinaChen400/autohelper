"""Attach measured model evidence to a review report, with explicit abstentions."""
import html
import json
from pathlib import Path
from analyze import analyze,word_error

folder = Path(__file__).resolve().parent/'sessions'/'original_video'
trial = json.loads((folder/'benchmark.json').read_text(encoding='utf-8'))
report = analyze(folder,reference=trial['target_text'])
report['model_trial'] = trial
report['limitations'].append('视频仅包含 Input 尾段，约 1.9 秒；不能用于可靠的音色投票')
for name in ('Input','A','B'):
    report['clips'][name]['transcript'] = trial['transcripts'][name]
    if name != 'Input':
        report['clips'][name]['target_word_error_rate'] = word_error(trial['target_text'],trial['transcripts'][name])
report['evaluation_draft']['Audio and Recording Quality'].update(reason='SigMOS 总体分 A≈2.96，B≈2.90，差距小且未校准；不据此自动投票。')
report['evaluation_draft']['Persona Likeness (Speaker Similarity)'].update(reason='Input 只有尾段，ECAPA 数值仅供实验，弃权。')
report['evaluation_draft']['Pronunciation Faithfulness'].update(reason='A/B 转写基本相同；dorsoventral 被转写成 dorsal ventral，可能是分词差异，不能据此认定错读。')
report['evaluation_draft']['Naturalness'].update(reason='尚未验证自然度评分模型，需人工回听。')
(folder/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
page = '<!doctype html><meta charset="utf-8"><title>原视频模型试验</title><style>body{font:17px sans-serif;max-width:1050px;margin:30px auto;background:#f6f7fa}section{background:white;padding:18px;margin:12px 0}audio{width:95%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:10px}</style>'
page += '<h1>原视频：独立模型试验</h1><p><b>Input 只录到尾段，不能据此作可靠音色判断。没有勾选、提交。</b></p><p>音频边界由画面播放状态与音轨静音间隙共同估计，请回听核对。</p>'
page += '<h2>页面目标文本</h2><p>'+html.escape(trial['target_text'])+'</p>'
for name,clip in report['clips'].items():
    page += f'<section><h2>{name}'+('（尾段，不完整）' if name=='Input' else '')+'</h2>'
    page += f'<audio controls src="{name}.wav"></audio><p>转写：{html.escape(clip["transcript"])}</p></section>'
page += '<h2>逐维度结论</h2><table><tr><th>维度</th><th>当前证据</th><th>选择</th></tr>'
for name,row in report['evaluation_draft'].items():
    page += f'<tr><td>{html.escape(name)}</td><td>{html.escape(row["reason"] or "证据不足，待人工判断")}</td><td>未选择</td></tr>'
page += '</table><h2>实测耗时（秒）</h2><pre>'+html.escape(json.dumps(trial['timings'],indent=2,ensure_ascii=False))+'</pre>'
page += '<p>处理时间不含音频本身播放时长、下载和界面操作。首次运行可能有初始化开销；不能由一题推断整体准确率。</p>'
(folder/'report.html').write_text(page,encoding='utf-8')
print(folder/'report.html')
