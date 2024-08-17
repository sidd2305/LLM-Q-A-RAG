# LLM Q&A RAG

**This Repository** is a powerful application designed for advanced question-answering using Retrieval-Augmented Generation (RAG) techniques. It integrates with Mistral's language models to provide contextually relevant answers from various sources, including URLs and uploaded files.

## Features

- **URL Processing**: Enter URLs of news articles or other sources to build a knowledge base.
- **File Upload**: Upload files in CSV, TXT, or PDF formats to use as a reference for generating answers.
- **Customizable Model**: Utilizes Mistral's language models with options to switch between different configurations.
- **Dynamic Q&A**: Interact with the application by querying the knowledge base to get accurate and relevant answers.

## Installation

To get started with the RAG Tool, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/sidd2305/RAG-LLM-Q-A-Assistant.git
    cd RAG-LLM-Q-A-Assistant
    ```

2. **Set Up Environment**:
    Create a virtual environment and install the required packages:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use: env\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Create Environment File**:
    Create a `.env` file in the root directory and add your Mistral API key:
    ```
    Mistral_API_KEY=your_Mistral_api_key
    ```

4. **Run the Application**:
    Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Open the App**:
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

2. **Select Mode**:
    - **Process URLs**: Enter up to 3 URLs of news articles or other sources.
    - **Upload File**: Upload a file in CSV, TXT, or PDF format.

3. **Query the Model**:
    Enter your question in the text input box and click "Submit" to get the answer from the processed data.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Mistral for providing the language models.
- Streamlit for the interactive app framework.
- LangChain for facilitating text processing and embeddings.

## Contact

For any questions or suggestions, please contact [Siddhanth Sridhar](mailto:siddhanth2305@gmail.com).

