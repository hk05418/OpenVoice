import argparse
from configparser import DEFAULTSECT
from contextlib import suppress
import os
import torch
import langid
import uvicorn
import shutil
import io

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, Response

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
# async def clone_voice(ref_audio: UploadFile = File(None), style: str = Form("default"), tts_text: str = Form()):
# async def clone_voice(tts_text: str = Form(), style: str = Form("default")):
async def clone_voice(tts_text, style="default"):

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
        target_se, audio_name = se_extractor.get_se(DEFAULT_REF_AUDIO, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        return OpenVoiceJSONResponse(400, f"[ERROR] Get target tone color error {str(e)}")

    # output_path 为None 时，返回音频数据
    audio_data = tts_model.tts(tts_text, output_path=None, speaker=style, language=language)
    # audio_io = io.BytesIO(audio_data)
    # return StreamingResponse(audio_io, media_type="audio/mpeg")
    # 1. 生成原始音频（临时保存）

    temp_wav_path = os.path.join(output_dir, "temp.wav")
    tts_model.tts(tts_text, output_path=temp_wav_path, speaker=style, language=language)

    # 2. 使用 tone color converter 转换音色
    converted_path = os.path.join(output_dir, "converted.wav")
    tone_color_converter.convert(
        audio_src_path=temp_wav_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=converted_path
    )

    # # 3. 返回转换后的音频数据
    # with open(converted_path, "rb") as f:
    #     audio_bytes = f.read()

    # return Response(content=audio_bytes, media_type="audio/mpeg")

    # 流式返回
    def iterfile():
        with open(converted_path, "rb") as f:
            while chunk := f.read(1024):
                yield chunk
    return StreamingResponse(iterfile(), media_type="audio/mpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10060)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)




