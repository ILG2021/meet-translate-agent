# Meet Translate Agent

会议实时语音翻译助手，基于pipecat案例程序创建：

https://github.com/pipecat-ai/pipecat/tree/main/examples/simple-chatbot

注意：此项目目前只能在linux和mac下运行

## 环境变量

```bash
cp env.example .env
```

然后申请相应的密钥，其中daily的申请地址是：
https://dashboard.daily.co/ 创建一个房间，房间链接贴到DAILY_SAMPLE_ROOM_URL，
需要绑定银行卡才能正常使用

google密钥的申请地址：
https://aistudio.google.com/apikey

## 运行

创建并激活虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

安装依赖:

```bash
pip install -r requirements.txt
```

运行:

```bash
python bot-gemini.py
```

打开房间链接，就可以通话并且使用ai实时语音翻译了