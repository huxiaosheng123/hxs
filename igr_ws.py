# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  本demo测试时运行的环境为：Windows + Python3.7
#  本demo测试成功运行时所安装的第三方库及其版本如下，您可自行利用pip安装：
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import sys


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret

        # 设置测试音频文件
        self.AudioFile = AudioFile
        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"ent": "igr", "aue": "raw", "rate": 16000}

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/igr'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/igr " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url


# 收到websocket消息的处理
def on_message(ws, message):
    list1 = []
    # file = open(r'C:\Users\hxs\Desktop\yuanshiresult.txt','a')
    try:
        print('11111111111',message)
        message = eval(message)
        res = message['data']['result']
        #print('222222222222',message['data']['result'])
        age_child = res['age']['child']
        age_middle = res['age']['middle']
        age_old = res['age']['old']
        if age_child > age_middle:
            if age_child > age_old:
                #print(age_child)
                list1.append('age_child')
            else:
                #print(age_old)
                list1.append('age_old')
        elif age_middle > age_child:
            if age_middle > age_old:
                #print(age_middle)
                list1.append('age_middle')
            else:
                #print(age_old)
                list1.append('age_old')
        elif age_old > age_child:
            if age_old > age_middle:
                #print(age_old)
                list1.append('age_old')
            else:
                #print(age_middle)
                list1.append('age_middle')
        gender_female = res['gender']['female']
        gender_male = res['gender']['male']
        if gender_male > gender_female:
            list1.append('gender_male')
        else:
            list1.append('gender_female')
        # print(age_child)
        # print(age_middle)
        # print(age_old)
        # print(gender_female)
        # print(gender_male)
        print(list1)
        file = open(r'F:\shengchengresult.txt', 'a')
        file.write(list1[0] + ' ' + list1[1]+' ' )
        list1.clear()
    except Exception as e:
        print("receive msg,but parse exception:", e)
        list1.append('None')
        list1.append('None')
        #print(list1)
        file = open(r'F:\shengchengresult.txt', 'a')
        file.write(list1[0] +' '+list1[1]+' ')
        list1.clear()

# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        frameSize = 5000  # 每一帧的音频大小
        # intervel = 0.04  # 发送音频间隔(单位:s)
        status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                # 文件结束
                if not buf:
                    status = STATUS_LAST_FRAME
                # 第一帧处理
                # 发送第一帧音频，带business 参数
                # appid 必须带上，只需第一帧发送
                if status == STATUS_FIRST_FRAME:

                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # 中间帧处理
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                # 最后一帧处理
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # 模拟音频采样间隔
                # time.sleep(intervel)
                # print('11111111111111',d)
        ws.close()

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    # 测试时候在此处正确填写相关信息即可运行
    path = r'F:\shengchengaudio.txt'
    # # fun1(path)
    with open(path) as f:
        line = f.readlines()
        for line1 in line:
            line1 = line1.replace('\\','/')
            print(line1)
            line1_part = line1.strip('\n').split('/')[2].split('.')[0]
            print(line1_part)
            file = open(r'F:\shengchengresult.txt', 'a')
            file.write(line1_part+'\n')
            wsParam = Ws_Param(APPID='5f7adbd0', APIKey='dc25c5e068a6a9ecabad15fd383f263e',
                               APISecret='02050824ce3c1ea6f8a375f0747768f2',
                               AudioFile=line1.strip('\n'))
            websocket.enableTrace(False)
            wsUrl = wsParam.create_url()
            ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
            ws.on_open = on_open
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            #sys.stdout = Logger(r"E:\PycharmProjects\Real-Time-Voice-Cloning\metrics\aa.txt")  # 保存到D盘

            time2 = datetime.now()

