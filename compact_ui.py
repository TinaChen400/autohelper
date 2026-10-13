"""Small, single-entry dashboard for the integrated assistant."""
import tkinter as tk
from tkinter import ttk


def build_dashboard(app):
    root=app.root
    for widget in root.winfo_children():widget.destroy()
    root.title('向日葵助手 · 自动答题')
    root.geometry('760x820')
    root.minsize(720,780)
    root.configure(background='#f3f5f9')
    style=ttk.Style(root)
    style.theme_use('clam')
    style.configure('.',font=('Microsoft YaHei UI',10),background='#f3f5f9',foreground='#25324a')
    style.configure('Card.TFrame',background='#ffffff')
    style.configure('Card.TLabel',background='#ffffff')
    style.configure('Title.TLabel',font=('Microsoft YaHei UI',20,'bold'))
    style.configure('Muted.TLabel',foreground='#657187')
    style.configure('Primary.TButton',font=('Microsoft YaHei UI',13,'bold'),padding=(24,14),
                    background='#315fe8',foreground='#ffffff',borderwidth=0)
    style.map('Primary.TButton',background=[('disabled','#a8b5d6'),('active','#244ed0')])
    style.configure('Stop.TButton',padding=(20,14),foreground='#a52e3b',background='#fceef0')
    style.configure('TButton',padding=(10,6))
    style.configure('Treeview',rowheight=34,background='#ffffff',fieldbackground='#ffffff')
    style.configure('Treeview.Heading',font=('Microsoft YaHei UI',10,'bold'))
    shell=ttk.Frame(root,padding=22);shell.pack(fill='both',expand=True)
    ttk.Label(shell,text='自动答题助手',style='Title.TLabel').pack(anchor='w')
    ttk.Label(shell,text='点一次开始，识别、评价、提交和下一题自动完成。',style='Muted.TLabel').pack(anchor='w',pady=(4,18))
    target=ttk.Frame(shell);target.pack(fill='x')
    ttk.Label(target,text='远程窗口').pack(side='left',padx=(0,10))
    app.combo=ttk.Combobox(target,state='readonly',width=36)
    app.combo.pack(side='left',fill='x',expand=True)
    ttk.Button(target,text='刷新',command=app.refresh).pack(side='left',padx=(8,0))
    actions=ttk.Frame(shell);actions.pack(fill='x',pady=(18,16))
    app.start_button=ttk.Button(actions,text='开始自动答题',style='Primary.TButton',command=app.run_integrated)
    app.start_button.pack(side='left',fill='x',expand=True)
    app.stop_button=ttk.Button(actions,text='停止 · Esc',style='Stop.TButton',command=app.stop)
    app.stop_button.pack(side='left',padx=(12,0))
    app.task_type=tk.StringVar(value='题型将自动识别')
    app.audio_status=tk.StringVar(value='等待开始')
    app.result_context=tk.StringVar(value='每道题独立保存答案与耗时。')
    app.human_status=tk.StringVar(value='选齐后等 3–5 秒自动 Submit；你可以修改，也可以提前提交。')
    app.model_duration=tk.StringVar(value='—')
    app.question_duration=tk.StringVar(value='—')
    app.question_stage=tk.StringVar(value='等待开始')
    app.status.set('准备就绪。确认上方窗口正确后，点击开始。')
    card=ttk.Frame(shell,style='Card.TFrame',padding=16);card.pack(fill='x')
    app.health_frame=tk.Frame(card,background='#fff2d6',padx=10,pady=8)
    app.health_title=tk.Label(app.health_frame,background='#fff2d6',foreground='#7a4300',
        font=('Microsoft YaHei UI',11,'bold'),anchor='w')
    app.health_title.pack(fill='x')
    app.health_detail=tk.Label(app.health_frame,background='#fff2d6',foreground='#7a4300',
        wraplength=580,justify='left',anchor='w',font=('Microsoft YaHei UI',10))
    app.health_detail.pack(fill='x',pady=(3,0))
    ttk.Label(card,text='当前进度',style='Card.TLabel',font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w')
    app.health_before=card.winfo_children()[-1]
    ttk.Label(card,textvariable=app.question_stage,style='Card.TLabel',foreground='#315fe8').pack(anchor='w',pady=(6,0))
    ttk.Label(card,textvariable=app.status,style='Card.TLabel',wraplength=580).pack(anchor='w',fill='x',pady=(8,12))
    metrics=ttk.Frame(card,style='Card.TFrame');metrics.pack(fill='x')
    for title,var in [('模型生成评价',app.model_duration),('本题总耗时',app.question_duration)]:
        column=ttk.Frame(metrics,style='Card.TFrame');column.pack(side='left',fill='x',expand=True)
        ttk.Label(column,text=title,style='Card.TLabel',foreground='#657187').pack(anchor='w')
        ttk.Label(column,textvariable=var,style='Card.TLabel',font=('Microsoft YaHei UI',18,'bold')).pack(anchor='w',pady=(4,0))
    ttk.Label(shell,textvariable=app.human_status,wraplength=620,style='Muted.TLabel').pack(fill='x',pady=(12,4))
    ttk.Label(shell,text='你可以在网页修改答案；鼠标操作时会等你停稳。按 Esc 随时停止。',
              style='Muted.TLabel',wraplength=620).pack(fill='x',pady=(0,12))
    details=ttk.Frame(shell)
    def toggle():
        if details.winfo_manager():
            details.pack_forget();toggle_button.configure(text='展开评价详情与工具');root.geometry('760x820')
        else:
            details.pack(fill='both',expand=True,pady=(10,0));toggle_button.configure(text='收起评价详情与工具');root.geometry('900x980')
    toggle_button=ttk.Button(shell,text='展开评价详情与工具',command=toggle)
    toggle_button.pack(anchor='w')
    notebook=ttk.Notebook(details);notebook.pack(fill='both',expand=True)
    result=ttk.Frame(notebook,padding=10);tools=ttk.Frame(notebook,padding=10);preview=ttk.Frame(notebook)
    notebook.add(result,text='评价结果');notebook.add(tools,text='设置与历史');notebook.add(preview,text='识别画面')
    ttk.Label(result,textvariable=app.task_type).pack(anchor='w')
    ttk.Label(result,textvariable=app.audio_status,wraplength=780).pack(fill='x',pady=6)
    app.rating_table=ttk.Treeview(result,columns=('dimension','result'),show='headings',height=5)
    app.rating_table.heading('dimension',text='维度');app.rating_table.heading('result',text='评价与依据')
    app.rating_table.column('dimension',width=110,stretch=False);app.rating_table.column('result',width=600)
    app.rating_table.pack(fill='both',expand=True)
    for key,title in [('quality','音质'),('speaker','音色相似度'),('pronunciation','发音忠实度'),('naturalness','自然度'),('overall','整体偏好')]:
        app.rating_table.insert('','end',iid=key,values=(title,'等待本题分析'))
    ttk.Label(result,textvariable=app.result_context,wraplength=780).pack(fill='x',pady=6)
    ttk.Label(tools,text='日常使用无需调整。设置从下一次开始生效。').pack(anchor='w',pady=(0,8))
    for title,command in [('查看历史题目',app.load_saved_result),('查看本题转写与耗时',app.show_transcripts),
                          ('核对保存的题目并继续',app.resume_remote)]:
        ttk.Button(tools,text=title,command=command).pack(anchor='w',pady=3)
    setting=ttk.Frame(tools);setting.pack(fill='x',pady=10)
    for index,(title,var) in enumerate([('补听停顿最少秒',app.pause_min),('最多秒',app.pause_max),('最多滚动段数',app.scroll_max)]):
        ttk.Label(setting,text=title).grid(row=index,column=0,sticky='w',pady=3)
        ttk.Entry(setting,textvariable=var,width=8).grid(row=index,column=1,padx=8)
    app.canvas=tk.Canvas(preview,background='#202b40',height=220,highlightthickness=0)
    app.canvas.pack(fill='both',expand=True)
    ttk.Label(preview,textvariable=app.summary,wraplength=780).pack(fill='x',padx=8,pady=6)
    app.refresh()


def show_health_alert(app,alert):
    if not alert:
        app.health_frame.pack_forget();return
    color,ink=('#fde9ec','#9a2033') if alert['level']=='error' else ('#fff2d6','#7a4300')
    app.health_frame.configure(background=color)
    app.health_title.configure(background=color,foreground=ink,text=alert['title'])
    app.health_detail.configure(background=color,foreground=ink,text=alert['detail']+'\n'+alert['action'])
    if not app.health_frame.winfo_manager():
        app.health_frame.pack(fill='x',pady=(0,12),before=app.health_before)
