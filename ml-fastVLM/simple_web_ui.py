#!/usr/bin/env python3
"""
Simple FastVLM Web UI
Just video stream + VLM output + prompt/token controls
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
import threading
from queue import Queue

# Fix Gradio bug
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

# Configuration
RTSP_URL = "rtsp://11QgE4OL:qrIW0btYlBUouIDZ@10.104.8.173:1000/live/ch0"
MODEL_NAME = "apple/FastVLM-0.5B"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Global state
model_loaded = False
tokenizer = None
model = None
image_processor = None
current_frame = None
current_response = "Waiting for analysis..."
frame_queue = Queue(maxsize=2)
response_queue = Queue(maxsize=1)
video_thread = None
video_active = False

def load_model():
    """Load the model once"""
    global tokenizer, model, image_processor, model_loaded
    
    if model_loaded:
        return tokenizer, model, image_processor
    
    print("📥 Loading model...")
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
    
    model_loaded = True
    print("✅ Model loaded!")
    return tokenizer, model, image_processor

def analyze_frame(frame, prompt, max_tokens):
    """Analyze a single frame"""
    try:
        tokenizer, model, image_processor = load_model()
        
        # Convert BGR to RGB
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
        
        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[pil_image.size],
                max_new_tokens=max_tokens,
                min_new_tokens=15,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Extract response - decode full output first
        full_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        prompt_decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0].strip()
        
        # Extract only the generated part
        input_token_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # If response seems incomplete or starts weirdly, try extracting from full output
        if not response or len(response) < 5 or response.startswith(('the image', 'Question:', 'Analysis:')):
            # Try to find where the actual response starts
            if full_output.startswith(prompt_decoded):
                response = full_output[len(prompt_decoded):].strip()
            else:
                # Look for common response markers
                for marker in ["Answer:", "Yes", "No", "The", "This", "I see", "In the"]:
                    if marker in full_output:
                        idx = full_output.find(marker)
                        if idx > len(prompt_decoded) // 2:
                            response = full_output[idx:].strip()
                            break
                if not response or response.startswith(('the image', 'Question:', 'Analysis:')):
                    response = full_output
        
        # Clean up response - remove common artifacts
        response = response.strip()
        
        # Remove leading artifacts
        while response and response[0] in (',', '.', ' ', '\n', '\t', ':'):
            response = response[1:].strip()
        
        # Remove common prefixes that shouldn't be there
        prefixes_to_remove = [
            "Analysis:",
            "Question:",
            "the image.",
            "the image",
            "Answer:",
        ]
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
                while response and response[0] in (',', '.', ' ', '\n', '\t', ':'):
                    response = response[1:].strip()
        
        # Final validation
        if not response or len(response) < 3:
            response = "No response generated"
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def video_capture_loop(rtsp_url):
    """Background thread to continuously capture video frames"""
    global current_frame, video_active

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"❌ Could not open RTSP stream: {rtsp_url}")
        return

    print("📺 Video capture started")
    try:
        while video_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            current_frame = frame.copy()

            # Put frame in queue (non-blocking, drop old frames)
            try:
                frame_queue.put(frame.copy(), block=False)
            except:
                pass  # Queue full, skip this frame

            time.sleep(0.03)  # ~30 FPS display

    except Exception as e:
        print(f"Video capture error: {str(e)}")
    finally:
        cap.release()
        print("📺 Video capture stopped")

def start_video_capture(rtsp_url):
    """Start video capture in background thread"""
    global video_thread, video_active

    if video_thread and video_thread.is_alive():
        return "Video capture already running"

    video_active = True
    video_thread = threading.Thread(
        target=video_capture_loop,
        args=(rtsp_url,),
        daemon=True
    )
    video_thread.start()
    return "✅ Video capture started!"

def stop_video_capture():
    """Stop video capture"""
    global video_active
    video_active = False
    return "⏹️ Video capture stopped"

def get_latest_frame():
    """Get the latest frame from queue"""
    frame = None
    while not frame_queue.empty():
        try:
            frame = frame_queue.get_nowait()
        except:
            break
    
    if frame is None:
        return None
    
    # Convert BGR to RGB for display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

def get_latest_response():
    """Get the latest response from queue"""
    response = current_response
    while not response_queue.empty():
        try:
            response = response_queue.get_nowait()
        except:
            break
    return response

def update_display():
    """Update function for Gradio"""
    frame = get_latest_frame()
    response = get_latest_response()
    
    # Return None for frame if not available (Gradio will keep previous)
    # But ensure response always has a value
    if response is None or response == "":
        response = "Waiting for analysis..."
    
    return frame, response

def initialize_display():
    """Initialize display and start video capture"""
    status = start_video_capture(RTSP_URL)
    frame, response = update_display()
    return status, frame, response

def analysis_loop(rtsp_url, prompt, max_tokens, frame_interval):
    """Background thread to analyze frames at intervals"""
    global current_response

    last_analysis_time = 0
    print("🧠 Analysis loop started")

    try:
        while video_active:
            current_time = time.time()

            # Analyze frame at intervals
            if current_time - last_analysis_time >= frame_interval and current_frame is not None:
                try:
                    response = analyze_frame(current_frame.copy(), prompt, max_tokens)
                    current_response = response
                    response_queue.put(response, block=False)
                    last_analysis_time = current_time
                except Exception as e:
                    current_response = f"Error: {str(e)}"
                    response_queue.put(current_response, block=False)

            time.sleep(0.1)  # Check every 100ms

    except Exception as e:
        print(f"Analysis loop error: {str(e)}")
    finally:
        print("🧠 Analysis loop stopped")

def start_analysis(rtsp_url, prompt, max_tokens, frame_interval):
    """Start the video analysis"""
    # Start video capture if not already running
    start_video_capture(rtsp_url)

    # Start analysis thread
    thread = threading.Thread(
        target=analysis_loop,
        args=(rtsp_url, prompt, int(max_tokens), float(frame_interval)),
        daemon=True
    )
    thread.start()
    return "✅ Analysis started!"

# Build Gradio UI
with gr.Blocks(title="FastVLM - Simple Video Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 FastVLM Video Analysis")
    gr.Markdown("Real-time video analysis with FastVLM")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_output = gr.Image(label="Video Stream", type="pil", height=600)
            response_output = gr.Textbox(
                label="VLM Output",
                lines=4,
                interactive=False,
                show_copy_button=True
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            rtsp_input = gr.Textbox(
                label="RTSP URL",
                value=RTSP_URL,
                lines=1
            )
            
            prompt_input = gr.Textbox(
                label="Prompt",
                value="Is shoplifting happening? Answer Yes or No, then describe what you see.",
                lines=3
            )
            
            max_tokens_input = gr.Slider(
                label="Max Tokens",
                minimum=10,
                maximum=100,
                value=50,
                step=5
            )
            
            frame_interval_input = gr.Slider(
                label="Analysis Interval (seconds)",
                minimum=0.5,
                maximum=5.0,
                value=1.0,
                step=0.1
            )
            
            start_button = gr.Button("▶️ Start Analysis", variant="primary", size="lg")
            status_output = gr.Textbox(label="Status", interactive=False)
    
    # Initialize on page load
    demo.load(
        fn=initialize_display,
        inputs=[],
        outputs=[status_output, video_output, response_output]
    )

    # Continuous display updates
    timer = gr.Timer(0.1)  # Update every 100ms
    timer.tick(
        fn=update_display,
        inputs=[],
        outputs=[video_output, response_output]
    )
    
    # Start button
    start_button.click(
        fn=start_analysis,
        inputs=[rtsp_input, prompt_input, max_tokens_input, frame_interval_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    print("🚀 Starting Simple FastVLM Web UI...")
    print("📺 Open http://localhost:7861 in your browser")
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, show_api=False)

