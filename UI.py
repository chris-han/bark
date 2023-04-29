from bark import SAMPLE_RATE, generate_audio
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav
import gradio as gr
import os 
import bark_infiniti

voicePath = "bark/assets/prompts";
files = os.listdir(voicePath)
npz_files = [f for f in files if f.endswith(".npz")]
npz_names = [os.path.splitext(f)[0] for f in npz_files]

print("\nStarting Suno-AI bark TTS WebUI\n")

def start(prompt, voice):
    print("\nStarting...")
    print(" Prompt : " + prompt)
    print(" Voice : " + (npz_names[voice] if voice is not None else "Default"))
    print()
    #prompt = '\n'.join(prompts)
    
    if voice == None:
        #audio_array,x = generate_audio(text=prompt)
        audio_array = bark_infiniti.generate_long_audio(text=prompt)
    else:   
        audio_array,x = generate_audio(text=prompt, history_prompt=npz_names[voice])
        audio_array = bark_infiniti.generate_long_audio(text=prompt, history_prompt=npz_names[voice])
        
    write_wav("audio.wav", SAMPLE_RATE, audio_array)
    return "audio.wav"

with gr.Blocks() as demo:

    gr.Title="Bark TTS WebUI",
    gr.Markdown("Bark TTS WebUI")
    
    with gr.Row():

        prompts = gr.Textbox(label="Prompt", placeholder="Enter your text here", lines=4)

    with gr.Row():
        voice = gr.Dropdown(npz_names, type="index", label="Voice", info="Select the voice")
        launch_button = gr.Button("Launch")   
  
    with gr.Column():
        output = gr.Audio(label="Result", type="numpy")
        
    launch_button.click(
        start,
        [prompts, voice],
        [output],
    )        

demo.launch()