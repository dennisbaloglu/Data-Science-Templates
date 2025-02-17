#!python
from transformers import pipeline

def main():
    # 1. Sentiment Analysis
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment_text = "I love using Hugging Face models! They make NLP so much fun."
    sentiment_result = sentiment_analyzer(sentiment_text)
    print("Sentiment Analysis Result:")
    print(sentiment_result, "\n")

    # 2. Text Generation using GPT-2
    text_generator = pipeline("text-generation", model="gpt2")
    prompt = "Once upon a time"
    generated_text = text_generator(prompt, max_length=50, num_return_sequences=1)
    print("Text Generation Result:")
    print(generated_text, "\n")

    # 3. Question Answering
    qa_pipeline = pipeline("question-answering")
    qa_input = {
        "question": "Who developed the Transformers library?",
        "context": "The Transformers library was developed by the team at Hugging Face."
    }
    qa_result = qa_pipeline(qa_input)
    print("Question Answering Result:")
    print(qa_result, "\n")

    # 4. Named Entity Recognition (NER)
    ner_pipeline = pipeline("ner", grouped_entities=True)
    ner_text = "Hugging Face Inc. is a company based in New York City."
    ner_result = ner_pipeline(ner_text)
    print("Named Entity Recognition (NER) Result:")
    print(ner_result, "\n")

    # 5. Summarization
    summarizer = pipeline("summarization")
    article = (
        "Hugging Face is a company that provides open-source libraries for natural language processing. "
        "Their Transformers library has become one of the most popular tools for building state-of-the-art "
        "machine learning models in NLP."
    )
    summary = summarizer(article, max_length=45, min_length=20, do_sample=False)
    print("Summarization Result:")
    print(summary, "\n")

    # 6. Translation (English to French)
    translator = pipeline("translation_en_to_fr")
    english_text = "Hugging Face provides state-of-the-art machine learning models."
    translation = translator(english_text, max_length=50)
    print("Translation (EN -> FR) Result:")
    print(translation, "\n")

    # 7. Fill-Mask (Masked Language Modeling)
    mask_pipeline = pipeline("fill-mask")
    masked_sentence = "The best way to predict the future is to [MASK] it."
    mask_result = mask_pipeline(masked_sentence)
    print("Fill-Mask Result:")
    for prediction in mask_result:
        print(f"Token: {prediction['token_str']}, Score: {prediction['score']:.4f}")

if __name__ == "__main__":
    main()
