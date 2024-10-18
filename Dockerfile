FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    xformers==0.0.27.post2 opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod pydub audio_separator moviepy \
    diffusers==0.27.2 transformers==4.39.2 accelerate==0.28.0 omegaconf==2.3.0 onnx==1.16.1 onnxruntime==1.18.0 onnxruntime-gpu==1.18.0 mediapipe==0.10.14 insightface==0.7.3 icecream==2.1.3 && \
    git clone https://github.com/fudan-generative-vision/hallo2 /content/hallo2 && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/CodeFormer/codeformer.pth -d /content/hallo2/pretrained_models/CodeFormer -o codeformer.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/CodeFormer/vqgan_code1024.pth -d /content/hallo2/pretrained_models/CodeFormer -o vqgan_code1024.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/audio_separator/Kim_Vocal_2.onnx -d /content/hallo2/pretrained_models/audio_separator -o Kim_Vocal_2.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/audio_separator/download_checks.json -d /content/hallo2/pretrained_models/audio_separator -o download_checks.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/audio_separator/mdx_model_data.json -d /content/hallo2/pretrained_models/audio_separator -o mdx_model_data.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/audio_separator/vr_model_data.json -d /content/hallo2/pretrained_models/audio_separator -o vr_model_data.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/face_analysis/models/1k3d68.onnx -d /content/hallo2/pretrained_models/face_analysis/models -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/face_analysis/models/2d106det.onnx -d /content/hallo2/pretrained_models/face_analysis/models -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/face_analysis/models/face_landmarker_v2_with_blendshapes.task -d /content/hallo2/pretrained_models/face_analysis/models -o face_landmarker_v2_with_blendshapes.task && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/face_analysis/models/genderage.onnx -d /content/hallo2/pretrained_models/face_analysis/models -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/face_analysis/models/glintr100.onnx -d /content/hallo2/pretrained_models/face_analysis/models -o glintr100.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/face_analysis/models/scrfd_10g_bnkps.onnx -d /content/hallo2/pretrained_models/face_analysis/models -o scrfd_10g_bnkps.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/facelib/detection_Resnet50_Final.pth -d /content/hallo2/pretrained_models/facelib -o detection_Resnet50_Final.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/facelib/detection_mobilenet0.25_Final.pth -d /content/hallo2/pretrained_models/facelib -o detection_mobilenet0.25_Final.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/facelib/parsing_parsenet.pth -d /content/hallo2/pretrained_models/facelib -o parsing_parsenet.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/facelib/yolov5l-face.pth -d /content/hallo2/pretrained_models/facelib -o yolov5l-face.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/facelib/yolov5n-face.pth -d /content/hallo2/pretrained_models/facelib -o yolov5n-face.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/hallo2/net_g.pth -d /content/hallo2/pretrained_models/hallo2 -o net_g.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/hallo2/net.pth -d /content/hallo2/pretrained_models/hallo2 -o net.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/realesrgan/RealESRGAN_x2plus.pth -d /content/hallo2/pretrained_models/realesrgan -o RealESRGAN_x2plus.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/sd-vae-ft-mse/config.json -d /content/hallo2/pretrained_models/sd-vae-ft-mse -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/sd-vae-ft-mse/diffusion_pytorch_model.safetensors -d /content/hallo2/pretrained_models/sd-vae-ft-mse -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/stable-diffusion-v1-5/unet/config.json -d /content/hallo2/pretrained_models/stable-diffusion-v1-5/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors -d /content/hallo2/pretrained_models/stable-diffusion-v1-5/unet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/wav2vec/wav2vec2-base-960h/config.json -d /content/hallo2/pretrained_models/wav2vec/wav2vec2-base-960h -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/wav2vec/wav2vec2-base-960h/feature_extractor_config.json -d /content/hallo2/pretrained_models/wav2vec/wav2vec2-base-960h -o feature_extractor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/resolve/main/wav2vec/wav2vec2-base-960h/model.safetensors -d /content/hallo2/pretrained_models/wav2vec/wav2vec2-base-960h -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/wav2vec/wav2vec2-base-960h/preprocessor_config.json -d /content/hallo2/pretrained_models/wav2vec/wav2vec2-base-960h -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/wav2vec/wav2vec2-base-960h/special_tokens_map.json -d /content/hallo2/pretrained_models/wav2vec/wav2vec2-base-960h -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/wav2vec/wav2vec2-base-960h/tokenizer_config.json -d /content/hallo2/pretrained_models/wav2vec/wav2vec2-base-960h -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/hallo2/raw/main/wav2vec/wav2vec2-base-960h/vocab.json -d /content/hallo2/pretrained_models/wav2vec/wav2vec2-base-960h -o vocab.json

COPY ./worker_runpod.py /content/hallo2/worker_runpod.py
WORKDIR /content/hallo2
CMD python worker_runpod.py