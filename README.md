# NLP Fundamentals & Labs Website

## Overview

This project is a static HTML website designed to present comprehensive notes on Natural Language Processing (NLP). It covers both classic and neural NLP techniques, supplemented with detailed analyses of practical lab modules. The content is structured for educational purposes, particularly aimed at students undertaking advanced studies in AI and NLP, such as Master's level courses.

The website aims to provide clear explanations, simplified examples, discussions on scalability, and insights from lab experiments to foster a thorough understanding of NLP concepts.

## Website Structure

The website is organized into several HTML pages, each dedicated to specific topics or lab modules:

### Core NLP Topics

- `index.html`: The main landing page, providing an overview and navigation to all sections.
- `language_models.html`: Covers classic N-Gram language models, Maximum Likelihood Estimation (MLE), and various smoothing techniques (Laplace, Add-k, Backoff, Interpolation, Kneser-Ney).
- `text_processing.html`: Details advanced text processing techniques including subword tokenization (like BPE), normalization strategies, and string similarity measures (Levenshtein distance). (This page was originally for classic text processing and later updated/complemented by `basic_text_processing_foundations.html` and `regex.html` for more foundational aspects from the lab notes).
- `sequence_labelling.html`: Explains sequence labelling tasks like Part-of-Speech (POS) tagging, Named Entity Recognition (NER), and delves into Conditional Random Fields (CRFs).
- `sparse_embeddings.html`: Discusses sparse vector representations such as TF-IDF, vector similarity using Cosine Similarity, and word association with Pointwise Mutual Information (PMI).
- `regex.html`: Focuses on Regular Expressions, covering core syntax, NLP applications (tokenization, IE, cleaning), Python implementation, limitations, and best practices (based on foundational notes).
- `basic_text_processing_foundations.html`: Covers fundamental text processing including word/sentence segmentation, text normalization (lowercasing, punctuation), stemming, and lemmatization (based on foundational notes).
- `word_embeddings.html`: Introduces neural word embeddings, contrasting them with sparse representations and detailing models like Word2Vec (SGNS), GloVe, FastText, and Dependency-Based Word Embeddings.
- `recurrent_neural_networks.html`: Covers Recurrent Neural Networks (RNNs), including Simple RNNs (Elman Networks), LSTMs, GRUs, and advanced architectures like Stacked and Bidirectional RNNs.
- `seq2seq_attention.html`: Details Sequence-to-Sequence (Seq2Seq) models, the encoder-decoder architecture, and the pivotal role of Attention Mechanisms.
- `transformer_architecture.html`: Explains the Transformer architecture, focusing on self-attention, multi-head attention, positional embeddings, feed-forward networks, Add & Norm layers, and causal masking.
- `transformer_models_pretraining.html`: Discusses key Transformer-based models (BERT, SpanBERT, RoFormer, Llama/Llama 2) and their pre-training objectives.
- `finetuning_advanced_llm.html`: Covers fine-tuning strategies (BERT, Llama 2-Chat with SFT, RLHF, GAtt), advanced prompting (CoT, Self-Consistency, SELF-REFINE), and knowledge augmentation (RAG, GENREAD).
- `nlp_tasks_applications.html`: Provides an overview of various NLP tasks (QA, NER, SRL, WSD, RE, Text Generation, MT, NLI, Dialogue Systems, Fact Checking, Coreference Resolution, KBP) and the application of different models.
- `evaluation_metrics_nlp.html`: Discusses evaluation metrics in NLP, with a focus on ROUGE for text generation tasks.

### NLP Lab Modules

- `lab_regex.html`: Detailed notes from Lab 1, covering Regex definition, purpose, fundamental syntax, common NLP tasks using Regex, Python implementation, and limitations/best practices with lab-specific insights.
- `lab_crf.html`: Detailed notes from Lab 2 on Conditional Random Fields, including their definition, advantages, linear-chain architecture, feature functions, IOB tagging, an in-depth analysis of the CRF lab experiments (Tasks 1-5 with feature/training differences and performance comparison), CRF variations, and uses/limitations.
- `lab_bert.html`: Detailed notes from Lab 3 on BERT, covering its Transformer encoder foundation, multi-head self-attention, input representation, pre-training objectives (MLM, NSP), tensor dimensions, fine-tuning process, common applications, advantages, limitations, and an interpretation of sample lab fine-tuning results.
- `lab_llama.html`: Detailed notes from Lab 4 on Llama models, discussing their decoder-only architecture, key components (SwiGLU, RoPE, RMSNorm, GQA), model sizes, tensor dimensions, usage (pre-trained, prompt engineering, fine-tuning), common applications, advantages, limitations, and an interpretation of sample lab interaction results.

## Technologies Used

- **HTML5**: For structuring the content of the web pages.
- **Tailwind CSS**: A utility-first CSS framework for styling the website and ensuring a responsive design. Loaded via CDN.
- **KaTeX**: For rendering mathematical formulas and equations beautifully. Loaded via CDN.
- **Google Fonts (Inter)**: Used for typography.
- **JavaScript**: For minor enhancements like mobile menu toggling and dynamic year updates in the footer.

## How to View

1. Ensure all HTML files (`index.html`, `language_models.html`, etc., and the `lab_*.html` files) are in the same directory.
2. Open `index.html` in any modern web browser (e.g., Chrome, Firefox, Safari, Edge).
3. Navigate through the topics using the navigation bar at the top of each page or the topic grids on the homepage.

## Content Sources

The content for this website is derived from three primary sources:

- "NLP Exam Preparation: Classic NLP" notes.
- "NLP Exam Preparation: Neural Models" notes.
- "NLP Labs Detailed Notes" covering Regex, CRF, BERT, and Llama labs.

## Notes for Students (Especially AI Master's Level)

This website is designed to be a comprehensive resource for understanding both the theoretical underpinnings and practical applications of various NLP techniques.

### Key Areas to Focus On

- **Conceptual Understanding**: Focus on *why* certain methods were developed and *how* they address the limitations of previous approaches.
- **Scalability**: Pay attention to the "Scalability Notes" within each section, as understanding how these techniques perform with large datasets and models is crucial.
- **Simplified Examples**: Use the provided examples to solidify your understanding of core concepts.
- **Lab Insights**: The lab module pages provide a deeper dive into the practical aspects, experimental setups, and interpretation of results for key NLP models and techniques. This is particularly important for understanding the impact of feature engineering, hyperparameter tuning, and model architecture choices.
- **Connections**: Try to draw connections between classic techniques and their evolution into more complex neural models. For instance, how the limitations of n-gram smoothing led to more sophisticated methods, or how attention mechanisms in Seq2Seq models paved the way for Transformers.

This project aims to be a helpful study aid. Good luck with your NLP endeavors!