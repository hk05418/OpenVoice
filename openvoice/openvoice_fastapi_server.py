import argparse
from configparser import DEFAULTSECT
from contextlib import suppress
import os
import torch
import langid
import uvicorn
import shutil

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')


# load speaker embeddings
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)

# This online demo mainly supports English and Chinese
supported_languages = ['zh', 'en']

# 默认参考音频
DEFAULT_REF_AUDIO = "resources/jp2.mp3"


app = FastAPI(title="OpenVoice Server API")

def OpenVoiceJSONResponse(code, err):
    return JSONResponse(status_code=code,
        content={"code": code, "error": err})

@app.get("/clone_voice")
@app.post("/clone_voice")
async def clone_voice(ref_audio: UploadFile = File(None), style: str = Form("default"), tts_text: str = Form()):

    # 1. 如果上传了 ref_audio，则保存到本地临时文件
    if ref_audio:
        ref_path = f"tmp/{ref_audio.filename}"
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(ref_audio.file, f)
    else:
        ref_path = DEFAULT_REF_AUDIO

    #
    # first detect the input language
    language_predicted = langid.classify(tts_text)[0].strip()
    print(f"Detected language:{language_predicted}")

    if language_predicted not in supported_languages:
        # 输入的文本语言不支持
        return JSONResponse(status_code=400, content={"code": 400, "error": f"{language_predicted} 不支持的语言，只支持中文和英文"})

    if language_predicted == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = 'Chinese'
        if style not in ["default"]:
            result = JSONResponse(
                status_code=400,
                content={"code": 400, "error": f"{language} style 只支持 default"}
            )
            return result

    else:
        tts_model = en_base_speaker_tts
        if style == 'default':
            source_se = en_source_default_se
        else:
            source_se = en_source_style_se
        language = 'English'

        if style not in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']:
            result = JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "error": f"style 只能为'default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly'"
                }
            )
            return result

        if len(tts_text) < 2 or len(tts_text) > 200:
            return OpenVoiceJSONResponse(400, f"输入文本长度{len(tts_text)}, 文本要大于2个字符，并且小于200个字符")

        try:
            target_se, audio_name = se_extractor.get_se(ref_path, tone_color_converter, target_dir='processed', vad=True)
        except Exception as e:
            return OpenVoiceJSONResponse(400, f"[ERROR] Get target tone color error {str(e)}")

        # output_path 为None 时，返回音频数据
        audio_data = tts_model.tts(tts_text, output_path=None, speaker=style, language=language)

        return audio_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10060)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)




