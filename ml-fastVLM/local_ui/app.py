"""Modern FastVLM Web Interface - Real-time Video Captioning"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import cv2
import gradio as gr
import torch
from PIL import Image
import numpy as np

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

DEFAULT_RTSP_FEED = "rtsp://11QgE4OL:qrIW0btYlBUouIDZ@10.104.8.173:1000/live/ch0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_REGISTRY: Dict[str, Dict[str, Optional[str]]] = {
    "FastVLM 0.5B (HuggingFace)": {
        "path": "apple/FastVLM-0.5B",
        "model_base": None,
        "hf_model": True,
    },
    "FastVLM 0.5B (Local)": {
        "path": "checkpoints/fastvlm_0.5b_stage3",
        "model_base": None,
        "hf_model": False,
    },
    "FastVLM 1.5B (Stage 3)": {
        "path": "checkpoints/fastvlm_1.5b_stage3",
        "model_base": None,
        "hf_model": False,
    },
    "FastVLM 7B (Stage 3)": {
        "path": "checkpoints/fastvlm_7b_stage3",
        "model_base": None,
        "hf_model": False,
    },
}

# Monkey patch to fix Gradio API info bug
import gradio_client.utils as client_utils
_original_json_schema_to_python_type = client_utils._json_schema_to_python_type

def _patched_json_schema_to_python_type(schema, defs=None):
    """Patched version that handles bool values in schema."""
    if isinstance(schema, bool):
        return "Any"
    try:
        return _original_json_schema_to_python_type(schema, defs)
    except (TypeError, ValueError) as e:
        if "bool" in str(e) or isinstance(schema, bool):
            return "Any"
        raise

client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type


def capture_frame(rtsp_url: Optional[str] = None, image: Optional[Image.Image] = None, video_frame: Optional[np.ndarray] = None) -> Image.Image:
    """Grab a single frame from RTSP feed, uploaded image, or webcam."""
    if image is not None:
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        return image
    
    if video_frame is not None:
        # Convert numpy array from video/webcam to PIL Image
        if isinstance(video_frame, np.ndarray):
            # Handle BGR from OpenCV
            if len(video_frame.shape) == 3 and video_frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
            return Image.fromarray(video_frame)
        elif isinstance(video_frame, str):
            # If it's a video file path, extract a frame
            cap = cv2.VideoCapture(video_frame)
            try:
                if cap.isOpened():
                    success, frame = cap.read()
                    if success and frame is not None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(rgb_frame)
            finally:
                cap.release()
    
    if not rtsp_url:
        raise ValueError("Either RTSP feed URL, an image, or webcam frame must be provided.")

    cap = cv2.VideoCapture(rtsp_url)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open RTSP feed: {rtsp_url}")
        success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError("Failed to read a frame from the feed.")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    finally:
        cap.release()


def build_prompt(prompt: str, conv_mode: str, config) -> str:
    """Construct the text prompt that includes the FastVLM image tokens."""
    qs = prompt or "Describe the scene."
    if getattr(config, "mm_use_im_start_end", False):
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class ModelManager:
    """Load FastVLM checkpoints lazily and cache them."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple] = {}

    def _resolve_path(self, registry_path: str) -> Path:
        return (PROJECT_ROOT / registry_path).expanduser()

    def get(self, model_key: str):
        if model_key in self._cache:
            return self._cache[model_key]

        config = MODEL_REGISTRY[model_key]
        is_hf_model = config.get("hf_model", False)
        
        disable_torch_init()
        
        if is_hf_model:
            # Load from HuggingFace using transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_path = config["path"]
            print(f"Loading model from HuggingFace: {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == "mps" else torch.float32,
                device_map={"": DEVICE.type} if DEVICE.type != "cuda" else "auto",
                trust_remote_code=True,
            )
            model.eval()
            
            # Get image processor from vision tower
            if hasattr(model, 'get_vision_tower'):
                vision_tower = model.get_vision_tower()
                if vision_tower is not None:
                    if not vision_tower.is_loaded:
                        vision_tower.load_model()
                    image_processor = vision_tower.image_processor
                else:
                    raise RuntimeError("Vision tower not found in model.")
            else:
                raise RuntimeError("Model does not have get_vision_tower method.")
            
            context_len = getattr(model.config, "max_position_embeddings", 2048)
        else:
            # Load from local checkpoint
            model_path = self._resolve_path(config["path"])
            if not model_path.exists():
                raise FileNotFoundError(
                    "Model checkpoint path is missing. "
                    f"Download the FastVLM stage and place it at {model_path}."
                )

            model_name = get_model_name_from_path(str(model_path))
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                str(model_path),
                config["model_base"],
                model_name,
                device=DEVICE.type,
            )
            model.eval()
            
            # Ensure vision tower is loaded and image_processor is available
            if hasattr(model, 'get_vision_tower'):
                vision_tower = model.get_vision_tower()
                if vision_tower is not None and not vision_tower.is_loaded:
                    vision_tower.load_model()
                if vision_tower is not None and image_processor is None:
                    image_processor = vision_tower.image_processor
        
        if image_processor is None:
            raise RuntimeError("Failed to load image processor. The model may not be properly configured.")
        
        # Move model to device if not already there
        if DEVICE.type == "mps" and next(model.parameters()).device.type != "mps":
            model = model.to(DEVICE)

        self._cache[model_key] = (
            tokenizer,
            model,
            image_processor,
            context_len,
        )
        return self._cache[model_key]


model_manager = ModelManager()


def run_inference(
    model_key: str,
    prompt: str,
    conv_mode: str,
    temperature: float,
    top_p: Optional[float],
    num_beams: int,
    feed_url: Optional[str] = None,
    uploaded_image: Optional[Image.Image] = None,
    video_frame: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, str, float]:
    """Capture a frame and return the FastVLM prediction with timing."""

    start_time = time.time()
    image = capture_frame(feed_url, uploaded_image, video_frame)
    tokenizer, model, image_processor, _ = model_manager.get(model_key)

    if image_processor is None:
        raise ValueError("Image processor is None. The model may not be properly loaded.")
    
    # Ensure image_processor has image_mean attribute if needed
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    if image_aspect_ratio == 'pad' and (not hasattr(image_processor, 'image_mean') or image_processor.image_mean is None):
        image_processor.image_mean = [0.0, 0.0, 0.0]

    prompt_text = build_prompt(prompt, conv_mode, model.config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE, non_blocking=True)

    # Process images
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.to(DEVICE)

    # Get stopping criteria
    conv = conv_templates[conv_mode]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else (conv.sep2 or conv.sep)
    # Ensure stop_str is not None and is a string
    if stop_str is None or not isinstance(stop_str, str):
        stop_str = conv.sep if conv.sep else "<|im_end|>"
    if not stop_str:  # Empty string check
        stop_str = "<|im_end|>"
    # Add eos_token to stopping criteria for better stopping
    eos_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token else None
    # Build keywords list, ensuring all are valid strings
    keywords_list = []
    if stop_str and isinstance(stop_str, str) and stop_str.strip():
        keywords_list.append(stop_str.strip())
    if eos_token and isinstance(eos_token, str) and eos_token.strip():
        keywords_list.append(eos_token.strip())
    # Ensure we have at least one valid keyword
    if not keywords_list:
        keywords_list = ["<|im_end|>"]
    # Remove duplicates while preserving order
    seen = set()
    keywords_list = [k for k in keywords_list if k not in seen and not seen.add(k)]
    stopping_criteria = KeywordsStoppingCriteria(keywords_list, tokenizer, input_ids)

    generation_kwargs = {
        "do_sample": temperature > 0.01,  # Only sample if temperature is meaningful
        "temperature": max(0.01, temperature),  # Ensure minimum temperature
        "num_beams": 1,  # Always use 1 beam for faster inference
        "max_new_tokens": 128,  # Increased back to get better responses
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
    }
    # Remove None values
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    if top_p is not None and top_p < 1.0:
        generation_kwargs["top_p"] = float(top_p)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            stopping_criteria=[stopping_criteria],
            **generation_kwargs,
        )
        
        # Extract only newly generated tokens
        input_token_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        
        # Decode only the generated tokens
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Debug: log what we got
        if len(answer) > 0:
            # Check if answer contains actual text (not just punctuation/special chars)
            text_chars = sum(1 for c in answer if c.isalnum() or c.isspace())
            if text_chars == 0:
                # No actual text, try alternative extraction
                full_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                prompt_decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0].strip()
                
                # Try to extract response after prompt
                if full_output.startswith(prompt_decoded):
                    answer = full_output[len(prompt_decoded):].strip()
                else:
                    # Look for assistant role marker
                    if conv.roles and len(conv.roles) > 1:
                        assistant_role = conv.roles[1]
                        if assistant_role in full_output:
                            # Find last occurrence of assistant role
                            parts = full_output.rsplit(assistant_role, 1)
                            if len(parts) > 1:
                                answer = parts[-1].strip()
                            else:
                                answer = full_output
                        else:
                            answer = full_output
                    else:
                        answer = full_output
        
        # Remove stop tokens and separators
        if stop_str:
            # Remove stop_str from anywhere
            answer = answer.replace(stop_str, "").strip()
            # Remove from end multiple times
            while answer.endswith(stop_str.strip()):
                answer = answer[:-len(stop_str.strip())].strip()
        
        # Remove eos token if present
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token and tokenizer.eos_token in answer:
            answer = answer.replace(tokenizer.eos_token, "").strip()
        
        # Remove pad token if present
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token and tokenizer.pad_token in answer:
            answer = answer.replace(tokenizer.pad_token, "").strip()
        
        # Remove role markers
        if conv.roles and len(conv.roles) > 1:
            role_marker = conv.roles[1]
            if role_marker:
                # Remove role marker at start
                if answer.startswith(role_marker):
                    answer = answer[len(role_marker):].strip()
                # Remove role marker with colon
                if answer.startswith(role_marker + ":"):
                    answer = answer[len(role_marker + ":"):].strip()
                # Remove role marker with space
                if answer.startswith(role_marker + " "):
                    answer = answer[len(role_marker + " "):].strip()
        
        # Clean up leading/trailing whitespace and punctuation
        answer = answer.strip()
        # Remove leading colons, newlines, spaces
        while answer and answer[0] in (":", "\n", " ", "\t"):
            answer = answer[1:].strip()
        
        # Remove trailing separators
        while answer and answer[-1] in ("\n", " ", "\t"):
            answer = answer[:-1].strip()
        
        # Final validation
        if not answer or len(answer.strip()) == 0:
            answer = "No response generated."
        else:
            # Check if answer has meaningful content (at least some alphanumeric characters)
            text_chars = sum(1 for c in answer if c.isalnum())
            if text_chars == 0:
                # Only punctuation/special chars - invalid
                answer = f"⚠️ Model generated invalid response (only punctuation). Try:\n- Lowering temperature to 0.1\n- Changing the prompt\n- Using a different model"
            elif text_chars < 3 and len(answer) > 10:
                # Very few text chars but long string - likely invalid
                answer = f"⚠️ Model generated mostly invalid characters. Try adjusting temperature or prompt."

    inference_time = time.time() - start_time
    return image, answer, inference_time


def process_continuous_video(
    model_key: str,
    video_input: str,
    prompt: str,
    conv_mode: str,
    temperature: float,
    top_p: Optional[float],
    num_beams: int,
    frame_delay: float,
    progress=gr.Progress(),
):
    """Process video frames continuously in real-time."""
    import time
    
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        yield None, "❌ Error: Could not open video source.", ""
        return
    
    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            try:
                image, answer, inference_time = run_inference(
                    model_key,
                    prompt,
                    conv_mode,
                    temperature,
                    top_p,
                    num_beams,
                    None,  # feed_url
                    pil_image,  # uploaded_image
                    None,  # video_frame
                )
                timing_info = f"⏱️ Frame {frame_count} | {inference_time:.2f}s"
                yield image, answer, timing_info
            except Exception as e:
                yield pil_image, f"❌ Error on frame {frame_count}: {e}", ""
            
            # Delay between frames
            time.sleep(frame_delay)
            
            if progress:
                progress((frame_count,), desc=f"Processing frame {frame_count}...")
    
    finally:
        cap.release()


def handle_request(
    model_key: str,
    feed_url: str,
    uploaded_image: Optional[Image.Image],
    video_input: Optional[str],
    continuous_mode: bool,
    prompt: str,
    conv_mode: str,
    temperature: float,
    top_p: Optional[float],
    num_beams: int,
    frame_delay: float = 1.0,
) -> Tuple[Optional[Image.Image], str, str]:
    """Handle the Gradio event and return the latest frame plus the model reply."""

    try:
        # Handle continuous video processing
        if continuous_mode and video_input:
            # This will be handled by the generator function
            return None, "🔄 Starting continuous processing...", ""
        
        # Handle single frame from video
        video_frame = None
        if video_input and not continuous_mode:
            cap = cv2.VideoCapture(video_input)
            if cap.isOpened():
                success, frame = cap.read()
                if success:
                    video_frame = frame
                cap.release()
        
        image, answer, inference_time = run_inference(
            model_key,
            prompt,
            conv_mode,
            temperature,
            top_p,
            num_beams,
            feed_url if feed_url else None,
            uploaded_image,
            video_frame,
        )
        timing_info = f"⏱️ Inference time: {inference_time:.2f}s"
        return image, answer, timing_info
    except Exception as exc:  # noqa: BLE001
        return None, f"❌ Error: {exc}", ""


def build_ui() -> gr.Blocks:
    """Construct the modern Gradio UI elements."""

    conv_options = list(conv_templates.keys())
    
    # Custom CSS for modern look
    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .model-selector {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            <div class="main-header">
                <h1>🚀 FastVLM: Real-time Vision Language Model</h1>
                <p>Efficient Vision Encoding for Vision Language Models | Powered by Apple</p>
            </div>
            """,
            elem_classes="main-header"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                model_selector = gr.Dropdown(
                    label="Model",
                    value="FastVLM 0.5B (HuggingFace)",
                    choices=list(MODEL_REGISTRY.keys()),
                    elem_classes="model-selector",
                    info="HuggingFace model loads directly from the cloud",
                )
                
                with gr.Tabs():
                    with gr.TabItem("🖼️ Upload Image"):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                    
                    with gr.TabItem("📹 Video/Webcam"):
                        video_input = gr.Video(
                            label="Video or Webcam",
                            sources=["webcam", "upload"],
                            format="mp4",
                        )
                        continuous_mode = gr.Checkbox(
                            label="🔄 Real-time Continuous Processing",
                            value=False,
                            info="Process frames continuously like WebGPU demo",
                        )
                        frame_delay = gr.Slider(
                            label="Frame Delay (seconds)",
                            minimum=0.1,
                            maximum=5.0,
                            step=0.1,
                            value=1.0,
                            visible=False,
                        )
                        continuous_mode.change(
                            lambda x: gr.update(visible=x),
                            inputs=[continuous_mode],
                            outputs=[frame_delay],
                        )
                    
                    with gr.TabItem("🌐 RTSP Stream"):
                        feed_input = gr.Textbox(
                            label="RTSP URL",
                            value="",
                            placeholder="rtsp://...",
                        )
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Describe what you see in detail.",
                    lines=2,
                )
                
                gr.Markdown("### 🎛️ Generation Parameters")
                
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.1,
                    info="Lower = more focused (recommended: 0.1-0.3), Higher = more creative",
                )
                
                top_p_slider = gr.Slider(
                    label="Top-p (Nucleus Sampling)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.95,
                    info="Controls diversity via nucleus sampling",
                )
                
                conv_mode_input = gr.Dropdown(
                    label="Conversation Template",
                    value="qwen_2",
                    choices=conv_options,
                    info="Template format for the conversation",
                )
                
                run_button = gr.Button("✨ Analyze", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                
                frame_output = gr.Image(
                    label="Input Image",
                    type="pil",
                    interactive=False,
                    height=400,
                )
                
                timing_output = gr.Textbox(
                    label="⏱️ Performance Metrics",
                    interactive=False,
                    visible=True,
                    value="Waiting for analysis...",
                    info="Shows inference time and frame processing stats",
                )
                
                response_output = gr.Textbox(
                    label="FastVLM Response",
                    lines=10,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="Response will appear here...",
                )
        
        # Set up event handlers
        run_button.click(
            handle_request,
            inputs=[
                model_selector,
                feed_input,
                image_input,
                video_input,
                continuous_mode,
                prompt_input,
                conv_mode_input,
                temperature_slider,
                top_p_slider,
                gr.Number(value=1, visible=False),  # num_beams
                frame_delay,
            ],
            outputs=[frame_output, response_output, timing_output],
        )
        
        # Continuous processing handler
        def start_continuous(model_key, video_input, prompt, conv_mode, temp, top_p, beams, delay):
            if video_input:
                yield from process_continuous_video(
                    model_key, video_input, prompt, conv_mode, temp, top_p, beams, delay
                )
            else:
                yield None, "❌ Please select a video source first.", ""
        
        continuous_mode.change(
            lambda x: gr.update(interactive=not x),
            inputs=[continuous_mode],
            outputs=[run_button],
        )
        
        # Auto-start continuous processing when enabled
        def auto_process(model_key, video_input, prompt, conv_mode, temp, top_p, delay, enabled):
            if enabled and video_input:
                yield from process_continuous_video(
                    model_key, video_input, prompt, conv_mode, temp, top_p, 1, delay
                )
        
        video_input.change(
            auto_process,
            inputs=[
                model_selector,
                video_input,
                prompt_input,
                conv_mode_input,
                temperature_slider,
                top_p_slider,
                frame_delay,
                continuous_mode,
            ],
            outputs=[frame_output, response_output, timing_output],
        )

    return demo


def main() -> None:
    """Launch the Gradio front-end."""

    demo = build_ui()
    demo.launch(server_port=7860, share=True, show_api=False)


if __name__ == "__main__":
    main()
