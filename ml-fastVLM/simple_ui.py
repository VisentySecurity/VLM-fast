#!/usr/bin/env python3
"""
Simple FastVLM Web Interface
Clean, minimal interface for RTSP stream analysis
"""

import cv2
import torch
import time
import gradio as gr
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# Configuration
DEFAULT_RTSP_URL = "rtsp://11QgE4OL:qrIW0btYlBUouIDZ@10.104.8.173:1000/live/ch0"
MODEL_NAME = "apple/FastVLM-0.5B"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Monkey patch for Gradio bug
import gradio_client.utils as client_utils
_original_json_schema_to_python_type = client_utils._json_schema_to_python_type
def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    try:
        return _original_json_schema_to_python_type(schema, defs)
    except (TypeError, ValueError) as e:
        if "bool" in str(e) or isinstance(schema, bool):
            return "Any"
        raise
client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

# Global model cache
_model_cache = {}

def load_model():
    """Load model once and cache it"""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"], _model_cache["image_processor"]
    
    print("📥 Loading model from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE.type == "mps" else torch.float32,
        device_map={"": DEVICE.type},
        trust_remote_code=True,
    )
    model.eval()
    
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor
    
    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    _model_cache["image_processor"] = image_processor
    
    print("✅ Model loaded!")
    return model, tokenizer, image_processor

def capture_frame(rtsp_url: str) -> Image.Image:
    """Capture a frame from RTSP stream"""
    cap = cv2.VideoCapture(rtsp_url)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from stream")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    finally:
        cap.release()

def analyze_frame(rtsp_url: str, prompt: str, temperature: float) -> tuple:
    """Analyze a frame from RTSP stream - optimized for speed"""
    try:
        start_time = time.time()
        
        # Load model
        model, tokenizer, image_processor = load_model()
        
        # Capture frame
        image = capture_frame(rtsp_url)
        
        # Build prompt
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        messages = [{"role": "user", "content": qs}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(DEVICE)
        
        # Process image
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.to(DEVICE)
        
        # Generate with minimal tokens for speed
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                max_new_tokens=32,  # Reduced for faster response
                temperature=0.0,  # Deterministic for speed
                do_sample=False,  # No sampling for speed
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Extract response
        input_token_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        inference_time = time.time() - start_time
        
        return image, response, f"⏱️ Inference time: {inference_time:.2f}s"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""

# Global state for continuous processing
_processing_active = False

def process_continuous(rtsp_url: str, prompt: str, temperature: float, frame_delay: float):
    """Process frames continuously in real-time - generator function"""
    import time
    global _processing_active
    
    _processing_active = True
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        yield None, f"❌ Error: Could not open RTSP stream: {rtsp_url}", "", "Error connecting"
        _processing_active = False
        return
    
    model, tokenizer, image_processor = load_model()
    frame_count = 0
    alert_count = 0
    
    try:
        while _processing_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue
            
            frame_count += 1
            start_time = time.time()
            
            # Convert frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Build prompt
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
            messages = [{"role": "user", "content": qs}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(DEVICE)
            
            # Process image
            image_tensor = process_images([pil_image], image_processor, model.config)[0]
            image_tensor = image_tensor.to(DEVICE)
            
            # Generate - optimized for speed
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half(),
                    image_sizes=[pil_image.size],
                    max_new_tokens=32,  # Reduced for faster response
                    temperature=0.0,  # Deterministic for speed
                    do_sample=False,  # No sampling for speed
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Extract response
            input_token_len = input_ids.shape[1]
            generated_ids = output_ids[:, input_token_len:]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            inference_time = time.time() - start_time
            
            # Check for shoplifting keywords
            response_lower = response.lower()
            shoplifting_keywords = ["yes", "taking", "bag", "coat", "pocket", "stealing", "removing", "placing", "putting"]
            is_suspicious = any(keyword in response_lower for keyword in shoplifting_keywords) and "no" not in response_lower[:10]
            
            if is_suspicious:
                alert_count += 1
                timing_info = f"🚨 ALERT #{alert_count} | Frame #{frame_count} | {inference_time:.2f}s | {time.strftime('%H:%M:%S')}"
                alert_status = f"🚨 ALERT #{alert_count}: Suspicious activity detected!"
                response = f"🚨 ALERT: {response}"
            else:
                timing_info = f"✅ Frame #{frame_count} | {inference_time:.2f}s | {time.strftime('%H:%M:%S')}"
                alert_status = f"✅ Monitoring... ({frame_count} frames analyzed)"
            
            yield pil_image, response, timing_info, alert_status
            
            # Wait before next frame
            time.sleep(frame_delay)
    
    finally:
        cap.release()
        _processing_active = False
        yield None, "⏹️ Processing stopped", "", "Stopped"

def build_ui():
    """Build the Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚀 FastVLM Real-time Video Analysis
            **Powered by Apple FastVLM-0.5B from HuggingFace | Like WebGPU Demo**
            """
        )
        
        with gr.Row():
            with gr.Column():
                rtsp_input = gr.Textbox(
                    label="RTSP Stream URL",
                    value=DEFAULT_RTSP_URL,
                    lines=2,
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Is someone taking items from a table and putting them into a bag, coat, or personal item? Answer: YES or NO, then briefly describe what you see.",
                    lines=3,
                    info="Optimized prompt for shoplifting detection",
                )
                
                gr.Markdown("### ⚙️ Settings")
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.0,
                    info="Set to 0.0 for fastest, deterministic responses",
                )
                frame_delay_slider = gr.Slider(
                    label="Frame Processing Delay (seconds)",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=1.0,
                    info="Delay between processing frames (lower = faster, recommended: 1.0s)",
                )
                
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Real-time", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                    single_btn = gr.Button("🔍 Single Frame", size="lg")
                
                running_state = gr.State(value=False)
            
            with gr.Column():
                image_output = gr.Image(
                    label="Live Video Frame",
                    type="pil",
                    height=400,
                    streaming=True,
                )
                timing_output = gr.Textbox(
                    label="Performance Metrics",
                    interactive=False,
                    value="Waiting to start...",
                )
                response_output = gr.Textbox(
                    label="FastVLM Response (Updates in Real-time)",
                    lines=12,
                    show_copy_button=True,
                    value="Response will appear here when processing starts...",
                )
                
                alert_output = gr.Textbox(
                    label="🚨 Alert Status",
                    interactive=False,
                    value="Monitoring...",
                    visible=True,
                )
        
        # Single frame analysis
        single_btn.click(
            analyze_frame,
            inputs=[rtsp_input, prompt_input, temperature_slider],
            outputs=[image_output, response_output, timing_output],
        )
        
        # Continuous processing
        def start_processing():
            """Start continuous processing"""
            global _processing_active
            _processing_active = True
        
        def stop_processing():
            """Stop continuous processing"""
            global _processing_active
            _processing_active = False
        
        # Start button - starts continuous processing
        start_btn.click(
            start_processing,
            outputs=[],
        ).then(
            process_continuous,
            inputs=[rtsp_input, prompt_input, temperature_slider, frame_delay_slider],
            outputs=[image_output, response_output, timing_output, alert_output],
        )
        
        # Stop button - stops processing
        stop_btn.click(
            stop_processing,
            outputs=[],
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting FastVLM Web Interface...")
    demo = build_ui()
    demo.launch(server_port=7860, share=False, show_api=False)

