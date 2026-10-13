"""Local listening report from measured results, without automatic votes."""
import argparse
import html
import json
from pathlib import Path

parser=argparse.ArgumentParser(); parser.add_argument('folder',type=Path); args=parser.parse_args()
p=args.folder
r=json.loads((p/'model_comparison.json').read_text(encoding='utf-8'))
escape=html.escape
players=''.join(f'<h2>{n}</h2><audio controls src="{n}.wav"></audio><p>{escape(r["transcripts"][n])}</p>' for n in ('Input','A','B'))
q=r['quality']; s=r['speaker_similarity']
natural=json.loads((p/'naturalness.json').read_text(encoding='utf-8')) if (p/'naturalness.json').exists() else None
feedback=json.loads((p/'human_feedback.json').read_text(encoding='utf-8')) if (p/'human_feedback.json').exists() else {}
quality_conclusion=('用户回听反馈：'+escape(str(feedback['quality']))+'。原始分数仅作诊断，不因分差直接选A。'
                    if feedback.get('quality') else '原始分数仅作诊断；尚未确认有明显缺陷，不能直接据分差选择A。')
natural_cells=(f'<td>{natural["scores"]["A"]["mos"]:.3f}</td><td>{natural["scores"]["B"]["mos"]:.3f}</td><td>UTMOSv2参考分；未校准，不自动投票</td>'
               if natural and all(n in natural['scores'] for n in ('A','B')) else '<td colspan="2">尚未评价</td><td>需要听音对照</td>')
body=f'''<!doctype html><meta charset="utf-8"><title>本次三段模型比较</title>
<style>body{{font:17px system-ui;max-width:960px;margin:32px auto;padding:0 20px;line-height:1.7}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:10px;text-align:left}}audio{{width:100%}}.note{{background:#fff3cf;padding:14px}}</style>
<h1>本次录音：模型比较草稿</h1><p class="note">只作人工对照，未勾选、未提交。三段边界依据页面进度及声音间隙人工核对；尚需回听确认完整性，B 没有结束后的停止图标记录。</p>
<p>页面目标正文：{escape(r['manifest']['target_text'])}</p>
<table><tr><th>维度</th><th>A</th><th>B</th><th>草稿结论</th></tr>
<tr><td>音质：SigMOS整体音质预测</td><td>{q['A']['MOS_OVRL']:.3f}</td><td>{q['B']['MOS_OVRL']:.3f}</td><td>{quality_conclusion}</td></tr>
<tr><td>音色：ECAPA相对Input余弦相似度</td><td>{s['A']:.3f}</td><td>{s['B']:.3f}</td><td>模型倾向 A；相似度不是准确率</td></tr>
<tr><td>发音忠实度</td><td colspan="2">正文转写相同；两段均未转写 uh</td><td>未发现正文差异；不能直接断定同等正确</td></tr>
<tr><td>自然度</td>{natural_cells}</tr>
<tr><td>整体偏好及主要因素</td><td colspan="2">暂不决定</td><td>等待你的听感，不能只由音质/音色代替</td></tr></table>
{players}<h2>本机实测耗时</h2><p>三段ASR、SigMOS、ECAPA推理合计 {sum(v for k,v in r['timings'].items() if k!='whisper_load'):.2f} 秒；Whisper加载 {r['timings']['whisper_load']:.2f} 秒。完整进程本次约9秒。非持续后台服务的最终延迟。</p>
<p>当前只捕获到页面顶部及音质选项；其余维度沿用已知页面结构，具体选项尚未从本题下方画面验证。</p>'''
(p/'comparison.html').write_text(body,encoding='utf-8')
print(str((p/'comparison.html').resolve()))
