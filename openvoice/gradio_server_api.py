import gradio as gr
import requests

API_URL = "http://192.168.3.2:10060/clone_voice"  # 如果你部署在远程服务器，请换成对应地址

def clone_voice_gradio(tts_text, style, ref_audio):
    if not tts_text:
        return "请输入要合成的文本", None

    files = {}
    data = {"tts_text": tts_text, "style": style}
    if ref_audio is not None:
        files["ref_audio"] = (ref_audio.name, ref_audio, "audio/mpeg")

    try:
        response = requests.post(API_URL, data=data, files=files)
        if response.status_code == 200:
            return "语音合成成功", response.content
        else:
            return f"[错误] {response.status_code}: {response.json().get('error')}", None
    except Exception as e:
        return f"[异常] {str(e)}", None


with gr.Blocks(title="OpenVoice 在线语音克隆") as demo:
    gr.Markdown("## 🗣️ OpenVoice 在线语音克隆演示")
    with gr.Row():
        with gr.Column():
            tts_text = gr.Textbox(label="请输入要合成的文本", lines=3, placeholder="支持中文或英文")
            style = gr.Dropdown(
                label="风格（仅英文支持）",
                choices=["default", "whispering", "shouting", "excited", "cheerful", "terrified", "angry", "sad", "friendly"],
                value="default"
            )
            ref_audio = gr.Audio(label="参考音频（可选）", type="filepath")
            btn = gr.Button("开始合成")

        with gr.Column():
            status = gr.Textbox(label="状态提示")
            output_audio = gr.Audio(label="合成结果", type="filepath")

    btn.click(fn=clone_voice_gradio, inputs=[tts_text, style, ref_audio], outputs=[status, output_audio])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
