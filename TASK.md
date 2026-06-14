# Multimodal Campus Orientation Assistant Assignment

## Learning Outcomes

### LO1

Innovate and design multi-modal chatbots that effectively process and respond to text, voice, and visual inputs, incorporate natural language processing and generation techniques to enhance conversational capabilities.

### LO2

Effectively communicate and present comprehensive ethical considerations and regulatory compliance strategies for multi-modal chatbot development, addressing issues related to data privacy, bias mitigation, and responsible AI practices.

### LO3

Effectively communicate and present comprehensive ethical considerations and regulatory compliance strategies for multi-modal chatbot development, addressing issues related to data privacy, bias mitigation, and responsible AI practices.

---

# Assessment Criteria

**Weighting:** 100%

**Word Count:** 3000 words

---

# Problem Statement / Case Study

You are tasked with creating a campus orientation assistant (For BSBI campus for example) that helps new students and visitors find their way around a university campus.

The assistant should understand user inputs through multiple modalities and provide accurate information and recommendations.

It will use:

* Computer Vision to recognize images of campus buildings and signage.
* Automatic Speech Recognition (ASR) to transcribe spoken queries into text.
* Natural Language Processing (NLP) to interpret and respond to typed or transcribed questions.

Your application must access or curate suitable datasets containing:

* Photographs of campus locations
* Synthetic or recorded voice queries
* Textual descriptions of facilities and events

You will train separate models for:

* Image classification or retrieval
* Speech transcription or intent classification
* Text question answering

These modalities should then be fused into a unified architecture that selects or retrieves the most relevant information and returns directions or event information to the user.

---

# Pipeline Architecture

## Architecture Diagram

```text
                         User inputs — three modalities
                ┌──────────────────┬──────────────────────┐
                │                  │                      │
                ▼                  ▼                      ▼
        ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
        │  Image input │  │   Voice input    │  │  Text input  │
        │Campus photo /│  │  Spoken query    │  │Typed question│
        │     sign     │  └────────┬─────────┘  └──────┬───────┘
        └──────┬───────┘           │                   │
               │           Modality-specific processing│
               ▼                   ▼                   │
        ┌──────────────┐  ┌──────────────────┐         │
        │ CLIP encoder │  │   Whisper ASR    │         │
        │   Frozen ·   │  │Frozen · local ·  │         │
        │  ViT-B/32    │  │    no API        │         │
        └──────┬───────┘  └────────┬─────────┘         │
               │                   │                   │
               ▼           Voice transcript + text → shared NLP
        ┌──────────────┐           │                   │
        │ FAISS index  │           └──────────┬─────────┘
        │   Cosine     │                      ▼
        │  similarity  │           ┌──────────────────────┐
        └──────┬───────┘           │      DistilBERT      │
               │                   │ Trained intent       │
               │                   │    classifier        │
               │                   └──────────┬───────────┘
               │        Semantic embeddings   │
               │                              ▼
               │                   ┌──────────────────────┐
               │                   │   Intent embedding   │
               │                   └──────────┬───────────┘
               │                              │
               ▼                              ▼
        ┌─────────────────────────────────────────────┐
        │             Multimodal fusion               │
        │           MLP · routing · masking           │
        └───────────────────┬──────────┬──────────────┘
                            │          │
                            │          ▼
                            │  ┌──────────────────────┐
                            │  │  Campus knowledge    │
                            │  │       base           │
                            │  │ JSON / SQLite ·      │
                            │  │   ≥15 records        │
                            │  └──────────┬───────────┘
                            │             │
                            ▼             ▼
                   ┌──────────────────────────────────┐
                   │        Response generation       │
                   │    KB record → formatted answer  │
                   └──────────────────────────────────┘
```

## Architecture Description

The system accepts three modalities in parallel:

- **Image input** (campus photo / sign): passed through a frozen CLIP ViT-B/32 encoder, then queried against a FAISS index using cosine similarity to retrieve the closest matching campus location.
- **Voice input** (spoken query): transcribed locally by a frozen Whisper ASR model (no external API). The resulting transcript feeds into the shared NLP path.
- **Text input** (typed question): enters the shared NLP path directly.

Both voice transcripts and typed text flow into a fine-tuned DistilBERT intent classifier, which produces a semantic intent embedding representing what the user is asking.

The CLIP/FAISS image retrieval result and the DistilBERT intent embedding are concatenated and passed into a **multimodal fusion layer** (MLP with routing and masking) that handles any combination of modalities — absent modalities are masked out rather than required.

The fusion layer queries the **campus knowledge base** (JSON or SQLite, ≥15 records) to look up the matching location record. A **response generation** step formats the KB record into a human-readable answer (location name, description, opening hours, events, directions).

---

# Technical Requirements

## 1. Environment Setup (10%)

Configure a suitable development environment for multimodal AI applications including:

* PyTorch and TorchVision for deep learning.
* Transformers (Hugging Face) for language models.
* OpenCV for image processing.
* SpeechRecognition and Whisper or alternative speech to text tools.
* Librosa for audio feature extraction.
* Streamlit or Flask for the user interface.
* Optionally HTML, CSS and JavaScript for a custom web UI.
* API keys for any cloud services if used, for example Google Speech to Text or external embeddings.
* Provide screenshots or terminal output demonstrating your setup and dependency installation.

---

## 2. Data Acquisition & Exploration (15%)

### Visual Data

* Select one or more publicly available datasets such as indoor scene recognition datasets, architectural imagery or campus map images, or curate your own dataset of campus photos.
* Load and display representative sample images and annotate them with their location or category, for example library, cafeteria, lecture hall.
* Plot class distributions such as the number of images per building type or area and compare similar versus dissimilar locations.

### Voice and Text Data

* Generate synthetic voice queries using free text-to-speech tools or record your own voice queries.

* Include requests such as:

  * "Where is the chemistry department?"
  * "Show me events at the student union today"
  * "Find a quiet study area near the cafeteria."

* Collect or create a small corpus of campus FAQs, facility descriptions and event schedules to train or fine tune a text-based model.

* Transcribe audio files, tokenize and explore text queries, and analyze vocabulary and intent categories.

### Campus Knowledge Base

Construct a structured knowledge base that your chatbot will query at inference time to return factual answers about the campus.

At a minimum, this should be:

* A JSON file or
* A small SQLite database

Containing records for each campus location, with fields for:

* Building name
* Category (e.g. library, cafeteria, lecture hall)
* Short description
* Opening hours
* GPS coordinates or relative map reference
* Upcoming events

You may invent realistic fictional data if a real campus dataset is not available.

Aim for at least 15–20 location records.

This knowledge base is not a trained model. It is a structured lookup resource that your NLP and fusion components will query to formulate a final response.

Document:

* How you designed its schema
* How your system retrieves the correct record given a recognized intent and entity

---

## 3. Preprocessing (15%)

### Image Pipeline

* Resize and normalize images.
* Apply augmentation techniques such as:

  * Random cropping
  * Rotations
  * Color jitter
* Convert images to tensors and create batch loaders.

### Audio Pipeline

* Convert audio to Mel frequency cepstral coefficients (MFCCs) or appropriate features using Librosa or a similar tool.
* Normalize audio features and pad sequences for batch processing.

### Text Pipeline

* Lowercase the text.
* Remove stop words.
* Apply tokenization.
* Perform train and validation splits.
* Consider advanced techniques such as stemming or lemmatization if beneficial.
* Include code snippets and sample outputs that illustrate your preprocessing steps.

---

## 4. Model Design (20%)

### Vision Model

* Explore using CLIP embeddings (openai/clip-vit-base-patch32, available freely via HuggingFace) with a vector search library such as FAISS for image-based retrieval of similar locations.

### Alternative Approaches

* Fine tune a pretrained convolutional neural network such as ResNet50 or EfficientNet on your campus images.
* Adapt the final classification layer to predict the building or service category.

### Practical Recommendation for the Assignment

#### First Choice: CLIP + FAISS (Retrieval-Based)

Use openai/clip-vit-base-patch32 from HuggingFace.

* Encode each knowledge base location as a text description.
* At query time, encode the uploaded image.
* Find the nearest text embedding using FAISS cosine similarity.

Advantages:

* Zero labelled image training data required.
* Fast on CPU/GPU.
* Most robust approach for a small campus dataset.

#### Second Choice (Optional): CLIP Fine-Tuned with Contrastive Loss

If you have collected at least 5–10 photos per location:

* Fine-tune CLIP's image encoder using contrastive loss.
* Pull campus-specific photo embeddings closer to corresponding text descriptions.

#### Third Choice: EfficientNet-B0

* Use EfficientNet-B0 (not B4 or higher).
* Freeze all layers except the final classifier.
* Use as feature extraction rather than full fine-tuning.

---

### Speech and Text Models

#### Whisper

Use Whisper (base or small variant) to transcribe voice input into text.

* Runs locally.
* No API key required.
* Produces accurate transcripts suitable for downstream NLP processing.

#### BERT / DistilBERT

Fine-tune a BERT or DistilBERT model on:

* Campus FAQs
* Synthetic queries

Tasks:

* Intent recognition
* Extractive question answering

Example intents:

* find_location
* ask_hours
* find_event

---

### Multimodal Fusion

Extract semantic embeddings from whichever modality or modalities are provided by the user.

Since real users typically submit:

* An image
* A voice clip
* A typed question

The fusion layer should:

* Treat each modality as independently sufficient.
* Allow any combination as optional enriching context.

Approaches:

* Concatenate available embeddings.
* Pad absent modalities with learned zero vectors.
* Mask absent modalities via attention mechanisms.

Pass the resulting joint representation into:

* A multilayer perceptron (MLP), or
* A lightweight transformer layer

Output:

* Recommendation
* Direction
* Answer

Discuss in your report:

* Single-modality routing
* Multi-modality routing
* Trade-offs involved

---

## 5. Training & Evaluation (20%)

### Trainable Components

CLIP and Whisper are frozen pretrained models and do not require training.

Train:

1. DistilBERT Intent Classifier
2. Fusion MLP

### Training Techniques

* Early stopping
* Learning rate scheduling
* Weight decay

### Monitoring

Track and visualize:

* Loss curves
* Accuracy per epoch

Tools:

* HuggingFace Trainer logging
* Matplotlib
* TensorBoard (optional)

### Evaluation Metrics

#### CLIP Vision Pipeline

* Top-1 Retrieval Accuracy
* Top-3 Retrieval Accuracy

#### DistilBERT

* Accuracy
* Precision
* Recall
* F1 Score

#### Whisper

* Word Error Rate (WER)

#### Fused System

* End-to-End Knowledge Base Retrieval Accuracy

Provide:

* Tables
* Graphs
* Critical analysis

Discuss:

* Performance gaps
* Trade-offs between modalities
* Dataset limitations

---

## 6. Deployment & User Testing (10%)

### Streamlit Application

Support:

1. Image Upload
2. Voice Recording / Audio Upload
3. Typed Text Queries

Output must display:

* Campus location name
* Description
* Opening hours
* Relevant events
* Map reference or directions

### Test Scenarios

Conduct at least five structured test scenarios.

Examples:

* Uploading a library photo
* Asking "Where is the student union?" by voice
* Typing "Is the cafeteria open on Sundays?"

Record:

* Inputs
* Outputs
* Correctness
* Failure cases

### Dockerization

Create:

* Dockerfile
* requirements.txt

Dockerfile must:

* Specify base Python image
* Install dependencies
* Copy application files
* Define Streamlit run command

### Optional

Use Ngrok to expose the application for demonstrations.

---

## 7. Ethical & Regulatory Considerations (10%)

### GDPR and Privacy

Discuss:

* Voice data handling
* Image data handling
* GDPR compliance
* Data minimization
* Anonymization techniques

### Bias Analysis

#### Visual Bias

* Over-representation of certain building types

#### Linguistic Bias

* Coverage of diverse phrasings
* Accessibility-related requests

#### Speech Bias

* Whisper WER variation across accents

---

# Evaluation or Grading Criteria

| Task                                | Weighting |
| ----------------------------------- | --------- |
| Environment Setup                   | 10%       |
| Data Acquisition & Exploration      | 15%       |
| Preprocessing                       | 15%       |
| Model Design                        | 20%       |
| Training & Evaluation               | 20%       |
| Deployment & User Testing           | 10%       |
| Ethical & Regulatory Considerations | 10%       |

---

# Submission Guidelines

* A 2500–3000 word report detailing your system design, implementation and evaluation.
* Upload submission as a single PDF or DOC file.
* Include architecture diagram.
* Include data exploration plots.
* Include knowledge base schema documentation.
* Include training and evaluation results.
* Include metrics tables.
* Include critical analysis.
* Use Harvard referencing.

Additional Deliverables:

* Working prototype (Colab or Python scripts)
* Campus knowledge base (JSON or SQLite)
* Streamlit demonstration
* Dockerfile
* requirements.txt
* Supplementary code and data files

Students are encouraged to explore additional reputable sources in the digital library beyond the list provided to strengthen their individual research and critical analysis skills.

The unit lecturer can help with recommending additional literature.

---

# Passing Criteria Examples

## Example 1

Pass Mark: 50

* Component 1 (50% weighting): 55
* Component 2 (50% weighting): 55

Final Average: 55

Result: Pass (both components passed and therefore the unit is passed)

---

## Example 2

Pass Mark: 50

* Component 1 (50% weighting): 80
* Component 2 (50% weighting): 30

Final Average: 55

Result: Fail (because although the average is above 50, one of the components is a failure)
