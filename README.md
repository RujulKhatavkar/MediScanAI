# MEDSCANAI ⛑️🤖

## Overview
MEDSCANAI is an innovative solution that leverages advanced language modeling and optical character recognition (OCR) to assist in medical document processing and information extraction. It integrates natural language generation with OCR capabilities to provide a comprehensive analysis of scanned medical documents, helping users interact with medical information in a conversational manner.

## Features
1. **Language Model Integration**: Utilizes the Mistral-7B-Instruct language model to generate natural language responses. This allows the system to answer questions related to the content of medical documents, such as medication effects or condition information.
2. **Optical Character Recognition (OCR)**: Extracts text from medical images and scanned documents, converting them into a format that can be further analyzed or queried.
3. **Interactive Chatbot**: Offers a conversational interface where users can interact directly with the system, asking questions about extracted medical information.
4. **Command Line Integration**: Provides command line tools to execute processes and receive live outputs, allowing seamless integration into existing workflows.
5. **Web scrapping**: Mining medical data from the interrnet inorder to fine tune using beautiful soup.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/username/medscanai.git
   ```
2. Install the required dependencies using `pip`:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running the Chatbot
To start the chatbot:
```sh
python chatbot_ai.py
```
This script loads the language model and runs a chatbot interface where users can ask questions related to medical topics.

here is an example of the LLM responding to a query by a user

![image](https://github.com/user-attachments/assets/10b0d9c0-972e-48e3-9a47-a3f11d0025ad)


### Running OCR Processing
To process medical documents using OCR:
```sh
python OCR.py --input <path_to_image>
```
This script extracts text from the provided image file, which can then be analyzed by the language model.

## Key Modules
- **`chatbot_ai.py`**: Implements the main chatbot logic, using a pre-trained language model for question answering.
- **`OCR.py`**: Implements OCR functionality to convert images of medical documents into text.

## Examples
- Ask about medication effects:
  ```
  User: Is Adderall XR harmful?
  Bot: Adderall XR may have several side effects, including...
  ```
- Extract text from an image:
  ```sh
  python OCR.py --input ./documents/prescription.png
  ```

## Requirements
- Python 3.8+
- Libraries:
  - `torch`
  - `transformers`
  - `opencv-python`

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

