# LipSyncing: Real-Time AI-Driven Lip Sync Web App

## Project Status

Currently, the system can divide incoming audio into small chunks and send each chunk to the AI model for processing. However, due to limited time, post-processing of these small chunk video results for seamless streaming is not yet implemented. As a result, while chunked inference works, the streaming of the generated video segments to the browser is not fully functional at this stage.

## Introduction

**LipSyncing** is a real-time web application that generates lip-synced talking head videos from a single image and live audio input. The system leverages state-of-the-art AI models to animate a still portrait in sync with user-provided speech, enabling applications in virtual avatars, video dubbing, and more.

### Methodology

- **Image & Audio Input:** Users upload a portrait image and provide audio (live or recorded).
- **Audio Chunking:** Audio is streamed in real time, chunked, and processed for low-latency inference.
- **AI Inference:** The [SadTalker](https://github.com/Winfredy/SadTalker) model is used to generate realistic talking head video segments from the image and audio.
- **Streaming Output:** Generated video segments are fragmented and streamed to the browser using Media Source Extensions (MSE) for smooth, low-latency playback.

### AI Model

- **SadTalker:** A cutting-edge deep learning model for speech-driven 3D facial animation. It takes a single image and audio as input and outputs a video of the image speaking the audio content, with realistic lip and facial movements.
- **Enhancement (Optional):** Optionally, the [GFPGAN](https://github.com/TencentARC/GFPGAN) face enhancer can be applied for higher-quality results.

### Backend

- **FastAPI:** Provides the main WebSocket API for real-time communication.
- **RabbitMQ:** Manages job queues for scalable, asynchronous AI inference.
- **Redis:** Used for fast state management, Pub/Sub messaging, and temporary storage of images and audio.
- **Worker Service:** A dedicated Python worker consumes jobs, runs SadTalker inference, fragments video, and publishes results.

### Frontend

- **Vanilla JavaScript:** The frontend is a single-page app using plain JavaScript for maximum compatibility and performance.
- **Media Source Extensions (MSE):** Enables low-latency, segment-based video playback in the browser.
- **WebSocket:** Real-time communication for sending images, audio, and receiving video segments.

---

## Quick Start with Docker Compose

1. **Clone the repository**
    ```bash
    git clone https://github.com/Kiet0712/LipSyncing.git
    cd LipSyncing
    ```

2. **Build and start all services**
    ```bash
    docker compose up --build
    ```

3. **Open the app**
    - Visit [http://localhost:8000/static/](http://localhost:8000/static/) in your browser.

4. **Stopping the app**
    ```bash
    docker compose down
    ```

## Downloading Model Weights

Before running the application, you need to download the required SadTalker and GFPGAN model weights.  
A helper script is provided for your convenience:

```bash
cd SadTalker/scripts
bash download_models.sh
```

This will download all necessary checkpoints and place them in the correct directories.

**Important:**  
After downloading, you also need to copy the `gfpgan` folder from inside the `SadTalker` directory to the root of your project (outside `SadTalker`). This is required so the backend can correctly locate the GFPGAN weights.

```bash
cp -r SadTalker/gfpgan ./gfpgan
```

> Make sure to place SadTalker model files in the correct directory before building the images.