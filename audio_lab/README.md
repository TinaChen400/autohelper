# 独立音频实验

## 原视频实测（2026-10-11）

已在独立环境下载并跑通 Whisper large-v3-turbo、微软 SigMOS V1、SpeechBrain ECAPA-TDNN。转写走 RTX 4080 Laptop CUDA FP16，SigMOS/ECAPA 走 CPU。额外依赖见 requirements_evaluation.txt；Windows CUDA DLL 由本进程局部加载，不更改点击环境。

原录屏共约 19.46 秒，开场 Input 已显示 19/20 秒，只能提取不足 2 秒尾段；不能据此可靠投票声音相似度。切片保存在 sessions/original_video，Input 0～1.9 秒、A 1.9～9.9 秒、B 9.9～16.65 秒；以视频状态转换和音轨间隙估计，需回听核对。音轨比画面按钮转换稍晚，不直接用画面时间硬切。

复测转写 A 约 0.22 秒、B 约 0.22 秒；三段转写、音质和说话人特征计算合计约 2.65 秒（不含模型加载/其他启动开销，不能推广到长音频）。首次 Whisper 推理约 21 秒、首次 SigMOS 特征处理约 18 秒，运行前必须预热。记录见 benchmark.json 及保留的历史结果。

A/B 转写基本一致；dorsoventral 的分词差异不作为错读判定。SigMOS 总体参考分约 A=2.96、B=2.90，差距小，不自动决定偏好。自然度尚未实现评分。报告 report.html 可回听，analysis.json 中所有评价选项仍为空。录音、模型权重和本地报告不提交 Git。

脚本：prepare_video.py 解码音轨/时间戳；benchmark_video.py 运行这份视频的固定切片试验；render_trial.py 生成带弃权说明的报告。这是一次离线试验，不是后台实时联动。

点击程序保持不动。此目录不导入 workflow、vision 或任何鼠标控制代码，单独使用 `.venv`。当前是录音/诊断/可选转写原型，没有音质、说话人相似度或自然度模型评分，也不会填写或提交远程页面。

## 先验证录音

1. 双击 `setup_audio_lab.bat` 安装独立环境依赖（需网络，未自动执行）。
2. 双击 `start_audio_lab.bat`，选择实际播放向日葵声音的输出设备。
3. 选 Input，点开始录音，等“正在录”提示后手动播放完整 Input，结束后停止。
4. 分别录 A、B。为避免当前自动点击流程混入重听片段，首轮验证时先停用自动流程，手动播放。
5. 每次启动新建一个 sessions 目录，三个文件是 Input.wav、A.wav、B.wav，不覆盖已存在文件。也可以把已有 PCM16 WAV 按这三个名称放进同一目录。

录的是所选设备系统输出，不是麦克风，也不是仅向日葵进程。通知或其他播放器声音会混入。回听确认三段对应且完整；不能只看文件存在。最长录制 10 分钟，停止保存后记录采集信息。尚未现场确认音频设备兼容性。

## 生成独立报告

在本目录执行：

```
.venv\Scripts\python.exe analyze.py sessions\本题目录
```

输出 report.html 可回听三段，analysis.json 保留录音诊断和待人工填写的评价。诊断指标不等于音质评分。

有本地模型后可转写（程序不自动下载模型）：

```
.venv\Scripts\python.exe analyze.py sessions\本题目录 --model D:\Models\faster-whisper-model --device cuda --reference sessions\本题目录\target.txt
```

target.txt 应是页面指定朗读的正文。Input 通常是音色参考，其正文可能与 A/B 不同，不能直接拿 Input 的转写检查 A/B 是否漏词。ASR 自身也会识别错；词差异只作为人工检查线索，不自动认定发音错误。GPU 使用需要匹配的 CUDA/cuDNN 依赖，未在本机安装和实测；CPU 模式可先验证。

## 后续评价模型和验收

先积累至少 20 个有完整录音和你逐维度判断的样本，记录分歧原因。发音忠实度需结合目标文本和听音，音色相似度需使用独立说话人嵌入模型，音质需语音质量模型，自然度及整体偏好需单独验证。不要让文本模型只看转写就断言音色或自然度。

建议每条结果保存 task_id、三段音频路径、目标文本、真实维度/选项、模型版本、各维度证据、你给出的答案和模型答案；允许弃权。当前草稿维度是样例，不代表所有页面固定相同，接入时以页面当题维度为准。

验证顺序：三段对应完整 → 转写可核对 → 分维度建议与你比较 → 本地模拟勾选 → 人工确认后再接远程填写。最终接入也应单独通过结果文件交换，不改已冻结的点击版本。不得直接据此自动提交。

参考：SoundCard https://github.com/bastibe/SoundCard；faster-whisper https://github.com/SYSTRAN/faster-whisper 。
