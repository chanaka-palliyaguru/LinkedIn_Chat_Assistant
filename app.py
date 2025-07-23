# Import necessary libraries
from groq import Groq
import gradio as gr
import os
from dotenv import load_dotenv
import logging
from pypdf import PdfReader
import soundfile as sf
import base64
import numpy as np
import io
import uuid

# Load environment variables from .env file
load_dotenv()
# Initialize environment variables
groq_key=os.environ.get("GROQ_API_KEY")

# Initialize Groq client globally for efficient API calls
if groq_key:
    try:
        client = Groq(api_key=groq_key)
        logging.info("Groq client initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Groq client: {e}")
else:
    logging.error("GROQ_API_KEY not found. Please ensure it's set in your environment variables or .env file.")

# JavaScript for handling voice activity detection (VAD) and audio processing in the browser
js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js";
  script2.onload = async () =>  {
    console.log("vad loaded")
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        console.log("Speech start detected")
        const showAnimatedMicButton = document.getElementById('show_animated_mic_button_id');
        if (showAnimatedMicButton) {
            console.log("Show animated mic button detected")
            showAnimatedMicButton.click(); 
        }
      },
      onSpeechEnd: (audio) => {
        console.log("Speech end detected")
        const float32Bytes = new Uint8Array(audio.buffer);
        let binaryString = '';
        const chunkSize = 16384; // Or 8192, 32768, etc. Adjust based on testing
        for (let i = 0; i < float32Bytes.length; i += chunkSize) {
            const chunk = float32Bytes.subarray(i, i + chunkSize);
            binaryString += String.fromCharCode(...chunk);
        }
        const base64Audio = btoa(binaryString);
        const vadTextInput = document.getElementById('audio_input_text'); 
        const hideAnimatedMicButton = document.getElementById('hide_animated_mic_button_id');    
        if (vadTextInput && hideAnimatedMicButton) {
            console.log("Text box and hide animated mic button detected")
            hideAnimatedMicButton.click();
            const vadTextInputElement = vadTextInput.querySelector('input[type="text"], textarea');
            if (vadTextInputElement) {
                console.log("Text box element detected")
                vadTextInputElement.value = base64Audio;
                vadTextInputElement.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
      }
    })
    myvad.start()
  }
  script1.onload = () =>  {
    console.log("onnx loaded") 
    document.head.appendChild(script2)
  };
}
"""

# CSS for styling Gradio components, especially for hiding elements
css = """
        .audio_input_text_css {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0,0,0,0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        .output_audio_css {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0,0,0,0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        .animated_mic_button {
            display: none !important;
        }
    """

# Read profile text from a file
with open("context/profile.txt", "r", encoding="utf-8") as f:
    profile = f.read()
logging.info("Profile text loaded from context/profile.txt.")

# Extract text from PDF for additional context
# reader = PdfReader("context/profile.pdf")
# pdf_profile = ""
# for page in reader.pages:
#     text = page.extract_text()
#     if text:
#         pdf_profile += text
# logging.info("Profile text extracted from context/profile.pdf.")

# Define the system prompt for the LLM, providing context and instructions
system_prompt = f""" You are a professional and engaging assistant representing Chanaka Palliyaguru. Your primary responsibility is to answer questions related to Chanaka's career, background, skills, and experience.
You have been provided Chanaka's profile. Use this information as your sole source to answer questions concisely and accurately. Prioritize direct answers over elaborate explanations.
Maintain a professional, engaging, and approachable tone, as if you are directly conversing with a potential client or future employer who has visited Chanaka's LinkedIn profile.
If you lack sufficient information, state directly: "I don't have that specific detail. Please contact Chanaka directly via LinkedIn messages for more information."
If a question is unrelated, state: "I apologize, but I can only provide information about Chanaka's professional background and experience."

\nProfile: {profile}
"""

# Function to show the animated microphone image and hide the static one
def showAnimatedMic():
    logging.info("Showing animated microphone.")
    return gr.update(visible=False), gr.update(visible=True)

# Function to hide the animated microphone image and show the static one
def hideAnimatedMic():
    logging.info("Hiding animated microphone.")
    return gr.update(visible=True), gr.update(visible=False)

# Function to process text messages from the user
def processMessage(message, chatbot_history):
     # Append the user's message to the chat history
     chatbot_history.append({"role": "user", "content": message})
     logging.info(f"User text message received: {message}")
     # Clear the input box and update the chatbot display
     yield "", chatbot_history
     
     llm_response_text = ""
     try:
        # Prepare messages for the LLM, including system prompt and conversation history
        conversation_history = [{"role": message["role"], "content": message["content"]} for message in chatbot_history]
        messages_for_llm = [{"role": "system", "content": system_prompt}] + conversation_history
        # Call Groq's chat completions API
        chat_completion = client.chat.completions.create(
            messages=messages_for_llm,
            model="llama-3.3-70b-versatile"
        )
        llm_response_text = chat_completion.choices[0].message.content
        logging.info(f"LLM text response generated: {llm_response_text}")
     except Exception as e:
        logging.error(f"Error during LLM chat completion for text message: {e}")
        llm_response_text = "I'm sorry, I'm having trouble connecting right now. Please try again in a moment."
     # Append the LLM's response to the chat history
     chatbot_history.append({"role": "assistant", "content": llm_response_text})
     # Update the chatbot display
     yield "", chatbot_history

# Function to process audio messages from the user
def processAudio(audio_base64, chatbot_history, session_data):
    # Generate a unique ID for temporary audio files
    unique_id = uuid.uuid4()
    temp_input_audio_file = f"audio/temp_input_audio_{unique_id}.wav"
    temp_output_audio_file = f"audio/temp_output_audio_{unique_id}.wav"
    sample_rate = 16000 # Define the sample rate for audio processing

    try:
        # Decode the Base64 audio string to bytes
        audio_bytes = base64.b64decode(audio_base64)
        # Convert bytes to a NumPy float32 array
        audio_np_array = np.frombuffer(audio_bytes, dtype=np.float32)
        # Write the NumPy array to a WAV file
        sf.write(temp_input_audio_file, audio_np_array, sample_rate)
        logging.info(f"User audio saved to {temp_input_audio_file}")
    except Exception as e:
        logging.error(f"Error saving audio file: {e}")
        # Clear the audio input and update chat history with an error message
        chatbot_history.append({"role": "assistant", "content": "There was an issue processing your audio input."})
        yield None, chatbot_history, session_data
        return

    # Transcribe audio using Groq's Speech-to-Text
    transcribed_text = ""
    try:
        # Open the temporary audio file for transcription
        with open(temp_input_audio_file, "rb") as file:
            # Call Groq's Speech-to-Text API
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
            # Check for high no-speech probability in the transcription
            if transcription.segments and len(transcription.segments) > 0:
                # Get no_speech_prob from the first segment
                no_speech_prob = transcription.segments[0].get('no_speech_prob', 0)
                if no_speech_prob > 0.7:
                    logging.info(f"High no-speech probability ({no_speech_prob}) detected. Returning empty transcription.")
                    # Add a message to the chatbot history indicating no speech
                    chatbot_history.append({"role": "assistant", "content": "I didn't detect any clear speech. Could you please try again?"})
                    yield None, chatbot_history, session_data # Clear the audio and update chatbot
                    os.remove(temp_input_audio_file) # Clean up temp file
                    return

            transcribed_text = transcription.text.strip()
            logging.info(f"Audio transcribed to text: {transcribed_text}")
    except Exception as e:
        logging.error(f"Error during audio transcription: {e}")
        chatbot_history.append({"role": "assistant", "content": "I had trouble understanding that. Could you please repeat?"})
        yield None, chatbot_history, session_data
        if os.path.exists(temp_input_audio_file):
            os.remove(temp_input_audio_file)
        return
    finally:
        if os.path.exists(temp_input_audio_file):
            os.remove(temp_input_audio_file)
    # Append the transcribed text to the chat history as a user message
    chatbot_history.append({"role": "user", "content": transcribed_text})
    # Yield to clear the audio input and update the chatbot
    yield None, chatbot_history, session_data

    # Send to LLM to process 
    llm_response_text = ""
    try:
        # Prepare messages for the LLM with the updated conversation history
        conversation_history = [{"role": message["role"], "content": message["content"]} for message in chatbot_history]
        messages_for_llm = [{"role": "system", "content": system_prompt}] + conversation_history
        # Call Groq's chat completions API
        chat_completion = client.chat.completions.create(
            messages=messages_for_llm,
            model="llama-3.3-70b-versatile"
        )
        llm_response_text = chat_completion.choices[0].message.content
        logging.info(f"LLM response to transcribed audio: {llm_response_text}")
    except Exception as e:
        logging.error(f"Error during LLM chat completion for audio input: {e}")
        llm_response_text = "I'm sorry, I'm having trouble connecting right now. Please try again in a moment."
    
    # Append the LLM's response to the chat history
    chatbot_history.append({"role": "assistant", "content": llm_response_text})
    yield None, chatbot_history, session_data

    # Convert LLM response to speech using Groq's Text-to-Speech 
    generated_audio = None
    try:
        # Call Groq's Text-to-Speech API to convert LLM response to audio
        response = client.audio.speech.create(
                model="playai-tts",
                voice="Fritz-PlayAI",
                input=llm_response_text,
                response_format="wav"
        )
        # Write the audio content to the temporary output audio file
        response.write_to_file(temp_output_audio_file)
        # Pass the file to Gradio output
        generated_audio = temp_output_audio_file
        # Store in session state to accessed during deletion
        session_data['output_audio_file'] = temp_output_audio_file
        logging.info("Text-to-Speech audio generated successfully.")
    except Exception as e:
        logging.error(f"Error during Text-to-Speech generation: {e}")
        # Update chatbot with a message about TTS failure
        chatbot_history.append({"role": "assistant", "content": "I couldn't generate speech for my last message."})
        yield None, chatbot_history, session_data
        return

    # Return the generated audio for playback, the updated chatbot history and the updated session
    yield generated_audio, chatbot_history, session_data

# Audio cleanup function that receives the session data
def cleanup_session_audio(session_data):
    file_to_delete = session_data.get('output_audio_file')
    if file_to_delete and os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        logging.info(f"Deleted temporary output audio file for session: {file_to_delete}")
    else:
        logging.error(f"Could not delete temporary output audio file for session: {file_to_delete} (file not found or path invalid)")
    # Clear the stored file after deletion
    session_data['output_audio_file'] = None
    # Clear audio player and return updated state
    return gr.update(value=None), session_data

# Gradio user interface definition
with gr.Blocks(theme=gr.themes.Soft(), js=js, css=css) as demo:
    # Initialize session state (a dictionary to store session-specific data)
    session_data = gr.State({})
    # Hidden Textbox to receive Base64 encoded audio from frontend JavaScript
    audio_text_input_box = gr.Textbox(elem_id="audio_input_text", elem_classes=["audio_input_text_css"])
    # Buttons to control visibility of microphone images, triggered by JavaScript
    show_animated_mic_button = gr.Button(elem_id="show_animated_mic_button_id", elem_classes=["animated_mic_button"])
    hide_animated_mic_button = gr.Button(elem_id="hide_animated_mic_button_id", elem_classes=["animated_mic_button"])
    chatbot = gr.Chatbot(label="Conversation", type="messages")
    with gr.Row():
        text_input_box = gr.Textbox(
            placeholder="Type your message here...",
            container=False,
            show_label=False,
            scale=7
        )
        send_button = gr.Button(
            "Send",
            variant="primary",
            scale=1
        )
        microphone_image = gr.Image(
            value="images/microphone_image.png", 
            type="filepath",
            interactive=False, 
            show_label=False,
            show_download_button=False,
            show_fullscreen_button=False,
            show_share_button=False,
            visible=True,
            height=40, width=40,
            scale=1
        )
        microphone_animated_image = gr.Image(
            value="images/microphone_animated_image.gif", 
            type="filepath",
            interactive=False, 
            show_label=False,
            show_download_button=False,
            show_fullscreen_button=False,
            show_share_button=False,
            visible=False, 
            height=40, width=40,
            scale=1
        )
    output_audio = gr.Audio(
            label="Output Audio",
            autoplay=True,
            visible=True,
            elem_classes=["output_audio_css"]
        )
    # Event listeners    
    send_button.click(
        fn=processMessage,
        inputs=[text_input_box, chatbot],
        outputs=[text_input_box, chatbot]
    )
    text_input_box.submit(
        fn=processMessage,
        inputs=[text_input_box, chatbot],
        outputs=[text_input_box, chatbot]
    )
    show_animated_mic_button.click(
        fn=showAnimatedMic,
        inputs=None,
        outputs=[microphone_image, microphone_animated_image]
    )
    hide_animated_mic_button.click(
        fn=hideAnimatedMic,
        inputs=None,
        outputs=[microphone_image, microphone_animated_image]
    )
    audio_text_input_box.input(
        fn=processAudio,
        inputs=[audio_text_input_box, chatbot, session_data],
        outputs=[output_audio, chatbot, session_data]
    )
    output_audio.stop( 
        fn=cleanup_session_audio,
        inputs=[session_data],
        outputs=[output_audio, session_data]
    )
# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
