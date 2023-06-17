<h1>Movie Review System</h1>
<hr size="1px">
<h2>Prerequisites</h2>
<hr size="0.5px">
<ol>
    <li>
        Before running the code, please ensure that you have the following installed: Python 3.7 or above, pip (Python package installer)
    </li>
    <li>
        Clone the repository: "git clone https://github.com/your-username/Movie_Review_System.git
        cd Movie_Review_System"
    </li>
    <li>
        Create a virtual environment (optional but recommended):
        python -m venv venv
        source venv/bin/activate
    </li>
    <li>
        Install the required dependencies: In the virtual environment :-
        pandas: pip install pandas
    sklearn: pip install scikit-learn
    nltk: pip install nltk
    string: No installation required (part of the Python standard library)
    pickle: No installation required (part of the Python standard library)
    streamlit: pip install streamlit
    transformers: pip install transformers
    translate: pip install translate
    certifi: pip install certifi
    </li>
    <li>
        Download the IMDb dataset: Download the IMDb dataset (CSV format) and save it in the project directory.
        The link :- "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resou rce=download
        "
    </li>
</ol>

<h2>
    Running the Application
</h2>
<ol>
    <li>Activate the virtual environment (if created): command line  :- "source venv/bin/activate
    </li>
    <li>
        Start the Streamlit application:- "streamlit run app.py
        "
    </li>
    <li>The application will start running and a local server will be created. You will see a URL displayed in the terminal (e.g., http://localhost:8501). Open that URL in your web browser.</li>
    <li>
        Enter your movie review in the provided text area.
        The sentiment analysis, summary generation, and translation options will be displayed.
    </li>
</ol>

<h2>Modelling of the Sentimental Analysis</h2>
<hr size="0.25">
<h6>Used the TF-IDF method to create the vectorizer and then used the linear SVC(Support Vector Classifier) to classify the sentiment , is is positive or negative</h6>
<h2>Summarizer</h2>
<hr size="0.25px">
<h6>Summarization Pipeline: The summarizer is accessed through the Hugging Face Transformers library's pipeline functionality. The pipeline allows for easy integration of the summarization model into the code. It abstracts away the complexities of loading the model, tokenization, and inference, providing a simple interface for generating summaries. The pipeline takes the input text and automatically applies the summarization model to generate a concise summary based on the specified parameters such as maximum and minimum length.
</h6>
<h2>Translator
</h2>
<hr size="0.25px">
<h6>translate Package: The translate package is a Python library that provides a simple interface for interacting with the Google Translate API. It allows you to specify the source language and the target language for translation, and provides functions to translate text strings.</h6>

