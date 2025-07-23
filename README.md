# LinkedIn_Chat_Assistant
An AI-powered LinkedIn chat assistant built with Groq and Gradio that acts as a digital version of myself. Users can interact with the assistant to learn about my career, professional background, skills, and experiences in a conversational format.

## Features
- Conversational AI: Interact with the AI assistant as if it was talking to another person.
- Text and Voice Input: Ask questions via text or by speaking directly to the assistant.
- Voice Activity Detection (VAD): Automatically detects speech segments for efficient audio processing.
- Real-time Transcription: Transcribes spoken questions into text using Groq's STT.
- Intelligent Responses: Generates accurate and concise answers based on the provided profile context using Groq's LLMs.
- Natural Voice Output: Converts AI responses into natural-sounding speech using Groq's TTS.
- Gradio Interface: User-friendly web interface for easy interaction.

## Technologies Used
- Gradio: For building the interactive web UI.
- Groq API:
  - Chat Completions: For LLM interactions (llama-3.3-70b-versatile).
  - Speech-to-Text: For transcribing user audio (whisper-large-v3-turbo).
  - Text-to-Speech: For generating audio responses (playai-tts).
- Python: The core programming language.
- JavaScript (Frontend): Utilizes onnxruntime-web and @ricky0123/vad-web for browser-side VAD and audio processing.

## Setup and Installation
Follow these steps to get the chat assistant up and running.
1. Clone the Repository
  - git clone https://github.com/chanaka-palliyaguru/LinkedIn_Chat_Assistant.git
  - cd LinkedIn_Chat_Assistant
2. Create and Activate the Virtual Environment
  - python -m venv venv
  - source `venv/bin/activate`  # On Windows: `venv\Scripts\activate`
3. Install Dependencies
  - pip install -r requirements.txt
4. Configure Environment Variables
  - Create a .env file in the root directory of your project and add your Groq API key: GROQ_API_KEY="your_groq_api_key_here"

## Running the Application
Once you have completed the setup, you can launch the Gradio interface.
  - run `python app.py`
The application will typically run on http://127.0.0.1:7860/ or a similar local address, which will be displayed in your      terminal. Open this URL in your web browser to start interacting with the AI chat assistant.

## Usage
- Text Input: Type your question into the text box and press "Send" or hit Enter.
- Voice Input: The system will automatically detect when you start speaking and when you finish. The animated microphone will indicate that it's listening. Once you stop speaking, your audio will be processed, and the AI will respond.

## Contributing
Feel free to fork this repository, submit pull requests, or open issues if you have suggestions or find bugs.

## License
This project is open-source and available under the MIT License.
