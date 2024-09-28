
# **AbleMinds AI**  

_A Generative AI-powered educational assistant for students with disabilities._

## **Overview**  
**AbleMinds AI** is an accessible educational assistant designed to help students with disabilities interact with learning materials in more inclusive ways. Using **Python** and advanced AI models like **Llama LLM** and **LangChain**, the system allows users to upload PDFs and ask questions about the content through various input methods, including **text**, **voice**, and even **sign language**. The system responds with answers as **text** and **voice output**, making learning more accessible and supportive for everyone.

### **Future Development**  
In the future, **AbleMinds AI** can be implemented into a **Raspberry Pi** to make it a fully integrated hardware solution, providing portable, affordable, and accessible learning assistance.

## **Features**
- **PDF Upload**: Users can upload their learning materials in PDF format.
- **Multiple Input Methods**:
  - Text input (keyboard)
  - Voice input (microphone)
  - Sign language input (future feature)
- **Multiple Output Methods**:
  - Text output
  - Voice output (speech synthesis)
- **Powered by LLM**: Uses **Llama LLM** to understand and generate responses.
- **Built with LangChain**: Efficient chaining and querying of large language models for interactive AI.

## **Tech Stack**
- **Frontend**: Streamlit for the user interface
- **Backend**: Python for processing input/output
- **Generative AI**: Llama LLM via LangChain for question answering
- **Voice Recognition & Synthesis**: SpeechRecognition and pyttsx3 for voice input/output
- **PDF Processing**: PyPDF2 for parsing and analyzing PDF content
- **Sign Language (Future Development)**: Integration with sign language recognition frameworks
- **Hardware (Future Development)**: Integration with Raspberry Pi for standalone, portable usage.

## **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/AbleMinds-AI.git
    cd AbleMinds-AI
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up necessary environment variables:
    - API keys for **speech recognition**, if using external services.
    - Model paths if you're running Llama LLM locally.

## **Usage**

1. Start the application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501/` to interact with the app.

3. **PDF Upload**: Upload your PDF file via the provided interface.

4. **Ask a Question**:
    - Type your question in the text box or use the microphone button for voice input.
    - (Future) Use a webcam for sign language input.

5. **Receive Responses**:
    - View the text-based response on the screen.
    - Listen to the answer via the text-to-speech system.

## **Screenshots**

### **1. Main Interface - PDF Upload**
![Main Interface](https://path-to-image.com/main-interface.png)

### **2. Text and Voice Input**
![Text and Voice Input](https://path-to-image.com/text-voice-input.png)

### **3. Answer Display**
![Answer Display](https://path-to-image.com/answer-display.png)

### **4. Future Raspberry Pi Implementation**
![Raspberry Pi Integration](https://path-to-image.com/raspberry-pi-integration.png)

## **Roadmap**
- [x] Text and voice input/output support
- [x] Add sign language recognition and input feature
- [x] Expand language model fine-tuning for more diverse educational content
- [ ] Implement cloud-based deployment for wider accessibility
- [ ] Hardware integration with Raspberry Pi for standalone usage

## **Contributing**
We welcome contributions to the project! Feel free to submit **pull requests** or create issues for any bugs or features you would like to see. Please follow our **contribution guidelines**.

## **Acknowledgments**
- **Llama LLM** for the powerful large language model.
- **LangChain** for streamlining the integration of generative AI.
- **Streamlit** for providing an intuitive framework to build the UI.
- **PyPDF2**, **SpeechRecognition**, and **pyttsx3** for enabling seamless PDF parsing and voice interaction.
