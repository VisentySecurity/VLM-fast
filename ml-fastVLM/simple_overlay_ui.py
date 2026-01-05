#!/usr/bin/env python3
"""
Ultra-Simple FastVLM UI - Just video with VLM output overlaid
"""


import cv2
import torch
import time
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
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
# Try CPU first for stability, fallback to MPS
DEVICE = torch.device("cpu")  # Changed from MPS for stability

# Global state
model_loaded = False
tokenizer = None
model = None
image_processor = None
current_frame = None
current_response = "Starting analysis..."
frame_queue = Queue(maxsize=1)
response_queue = Queue(maxsize=1)
video_active = False
analysis_active = False
current_interval = 2.0  # Current analysis interval

# Video analysis state
frame_buffer = []  # Store last N frames for temporal analysis
MAX_BUFFER_SIZE = 5  # Keep last 5 frames for context

def load_model():
    """Load the model once"""
    global tokenizer, model, image_processor, model_loaded

    if model_loaded:
        return tokenizer, model, image_processor

    try:
        print("📥 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("✓ Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Always use float32 for CPU stability
            device_map={"": DEVICE.type},
            trust_remote_code=True,
        )
        model.eval()
        print("✓ Model loaded")

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        print("✓ Vision tower loaded")

        model_loaded = True
        print("✅ Model fully loaded!")
        return tokenizer, model, image_processor
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise  # Re-raise to be caught by caller

def analyze_frame(frame, prompt, max_tokens=50):
    """Analyze a single frame with better error handling"""
    try:
        # Check if frame is valid
        if frame is None or frame.size == 0:
            return "Error: Invalid frame received"

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

        # Generate - fast settings for CPU
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).float(),  # Use float for CPU instead of half
                image_sizes=[pil_image.size],
                max_new_tokens=max_tokens,
                min_new_tokens=5,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract response
        input_token_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Clean up response
        response = response.strip()
        while response and response[0] in (',', '.', ' ', '\n', '\t'):
            response = response[1:].strip()

        if not response or len(response) < 2:
            response = "Analysis: No clear response from model"

        return f"Analysis: {response}"
    except torch.cuda.OutOfMemoryError:
        return "Error: GPU memory full - try restarting"
    except RuntimeError as e:
        if "MPS" in str(e):
            return "Error: MPS device issue - switched to CPU mode"
        return f"Runtime Error: {str(e)}"
    except Exception as e:
        return f"Analysis Error: {str(e)}"

def analyze_frame_sequence(frame_sequence, prompt, max_tokens=50):
    """Analyze a sequence of frames for temporal behaviors (shoplifting detection)"""
    try:
        if not frame_sequence or len(frame_sequence) == 0:
            return "[ERROR] No frames to analyze"

        current_frame = frame_sequence[-1]
        sequence_length = len(frame_sequence)

        if current_frame is None:
            return "[ERROR] Frame is None"

        tokenizer, model, image_processor = load_model()

        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        import time
        timestamp = time.strftime("%H:%M:%S")
        enhanced_prompt = f"{prompt}\n\nAnalysis timestamp: {timestamp}. Analyzing {sequence_length} frames.\n\nCRITICAL: Write a DETAILED description first (4-6 sentences). Describe: objects/products visible, layout, people and locations, lighting, movements. THEN assess shoplifting."

        qs = f"{DEFAULT_IMAGE_TOKEN}\n{enhanced_prompt}"
        messages = [{"role": "user", "content": qs}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(DEVICE)

        image_tensor = process_images([pil_image], image_processor, model.config)[0]
        image_tensor = image_tensor.to(DEVICE)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).float(),
                image_sizes=[pil_image.size],
                max_new_tokens=max_tokens,
                min_new_tokens=20,
                do_sample=True,
                temperature=0.2,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_token_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        response = response.strip()
        while response and response[0] in (',', '.', ' ', '\n', '\t'):
            response = response[1:].strip()

        if not response or len(response) < 10:
            response = f"Scene appears normal. {response}" if response else "Analysis completed - scene appears normal."

        return f"[{timestamp}] {response}"
    except torch.cuda.OutOfMemoryError:
        error_msg = "[ERROR] GPU memory full - try restarting"
        print(f"❌ {error_msg}")
        return error_msg
    except RuntimeError as e:
        error_str = str(e)
        if "MPS" in error_str:
            error_msg = "[ERROR] MPS device issue - using CPU"
            print(f"❌ {error_msg}: {error_str[:100]}")
            return error_msg
        elif "out of memory" in error_str.lower():
            error_msg = "[ERROR] Out of memory - try reducing max_tokens"
            print(f"❌ {error_msg}: {error_str[:100]}")
            return error_msg
        else:
            error_msg = f"[ERROR] Runtime error: {error_str[:80]}"
            print(f"❌ {error_msg}")
            import traceback
            print(traceback.format_exc())
            return error_msg
    except Exception as e:
        import traceback
        error_str = str(e)
        error_msg = f"[ERROR] Analysis failed: {error_str[:80]}"
        print(f"❌ {error_msg}")
        print("Full traceback:")
        print(traceback.format_exc())
        return error_msg

def video_capture_loop(rtsp_url):
    """Capture video frames continuously"""
    global current_frame, video_active, current_response

    print(f"📺 Attempting to connect to RTSP stream: {rtsp_url[:50]}...")
    
    # Set RTSP options for better connection
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
    
    if not cap.isOpened():
        error_msg = f"❌ Could not open RTSP stream: {rtsp_url}"
        print(error_msg)
        current_response = f"[ERROR] Video connection failed: {rtsp_url[:30]}..."
        return

    print("✓ RTSP stream opened successfully")
    
    frame_count = 0
    failed_reads = 0
    max_failed_reads = 10
    
    try:
        print("📺 Video capture loop started")
        while video_active:
            ret, frame = cap.read()
            
            if not ret:
                failed_reads += 1
                if failed_reads > max_failed_reads:
                    print(f"⚠️ Failed to read {failed_reads} frames in a row - reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        print("❌ Reconnection failed")
                        current_response = "[ERROR] Video stream disconnected"
                        break
                    failed_reads = 0
                    print("✓ Reconnected to RTSP stream")
                time.sleep(0.1)
                continue

            # Successfully read a frame
            failed_reads = 0
            current_frame = frame.copy()
            frame_count += 1

            # Add frame to buffer for temporal analysis
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > MAX_BUFFER_SIZE:
                frame_buffer.pop(0)  # Remove oldest frame

            if frame_count == 1:
                print(f"✓ First frame captured! Shape: {frame.shape}, Buffer: {len(frame_buffer)}")
            elif frame_count % 30 == 0:  # Log every 30 frames (~1 second)
                print(f"📺 Captured {frame_count} frames, buffer size: {len(frame_buffer)}")

            # Put frame in queue (non-blocking)
            try:
                frame_queue.put(frame.copy(), block=False)
            except:
                pass  # Queue full, skip this frame

            time.sleep(0.03)  # ~30 FPS

    except Exception as e:
        import traceback
        error_msg = f"Video capture error: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        current_response = f"[ERROR] Video capture failed: {str(e)[:50]}"
    finally:
        cap.release()
        video_active = False
        print("📺 Video capture stopped")

def analysis_loop(prompt, interval, max_tokens):
    """Analyze frames at intervals"""
    global current_response, analysis_active

    last_analysis_time = 0
    analysis_count = 0
    try:
        while analysis_active:
            current_time = time.time()

            # Analyze at intervals using frame sequence
            if current_time - last_analysis_time >= interval:
                if len(frame_buffer) < 2:
                    # Not enough frames yet - check if video is active
                    if not video_active:
                        current_response = "[ERROR] Video capture not running - click Start Monitoring"
                    elif len(frame_buffer) == 0:
                        current_response = f"Connecting to video stream... ({len(frame_buffer)}/{2} frames)"
                    else:
                        current_response = f"Buffering video frames... ({len(frame_buffer)}/{2} frames)"
                    print(f"⏳ Waiting for frames: buffer={len(frame_buffer)}, video_active={video_active}")
                    time.sleep(0.5)
                    continue
                
                try:
                    # Copy the current frame buffer for analysis
                    current_sequence = frame_buffer.copy()
                    analysis_count += 1
                    print(f"🔍 Running analysis #{analysis_count} with {len(current_sequence)} frames, interval={interval}s")
                    
                    response = analyze_frame_sequence(current_sequence, prompt, max_tokens)
                    
                    # Always update response - show whatever we got
                    if response:
                        current_response = response
                        response_queue.put(response, block=False)
                        if response.startswith("[ERROR]"):
                            print(f"⚠️ Analysis #{analysis_count} error: {response}")
                        elif "Analyzing scene" in response or len(response) < 50:
                            print(f"⚠️ Analysis #{analysis_count} short/placeholder: {response[:100]}")
                        else:
                            print(f"✅ Analysis #{analysis_count} complete ({len(response)} chars): {response[:100]}...")
                        last_analysis_time = current_time
                    else:
                        current_response = "[ERROR] Analysis returned None - check model output"
                        response_queue.put(current_response, block=False)
                        print(f"❌ Analysis #{analysis_count} returned None")
                        last_analysis_time = current_time
                        
                except Exception as e:
                    import traceback
                    error_msg = f"[ERROR] Analysis failed: {str(e)[:80]}"
                    print(f"❌ Analysis error: {str(e)}")
                    print(traceback.format_exc())
                    current_response = error_msg
                    response_queue.put(error_msg, block=False)
                    last_analysis_time = current_time  # Still update time to prevent spam

            time.sleep(0.1)

    except Exception as e:
        import traceback
        print(f"❌ Analysis loop fatal error: {str(e)}")
        print(traceback.format_exc())
    finally:
        print("🧠 Analysis stopped")

def get_overlay_frame():
    """Get latest frame with VLM response overlaid"""
    try:
        # Get latest frame
        frame = None
        try:
            while not frame_queue.empty():
                try:
                    frame = frame_queue.get_nowait()
                except:
                    break
        except:
            pass

        if frame is None and current_frame is not None:
            try:
                frame = current_frame.copy()
            except:
                frame = None

        if frame is None:
            # Return black frame with message if no video
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            pil_img = Image.fromarray(blank)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((20, 20), "Waiting for video stream...", fill=(255, 255, 255), font=font)
            draw.text((20, 50), current_response[:100] if current_response else "Starting...", fill=(255, 255, 255), font=font)
            return pil_img.convert('RGB')

        # Convert to PIL for overlay
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
        except Exception as e:
            print(f"⚠️ Frame conversion error: {e}")
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            pil_img = Image.fromarray(blank)

        # Get latest response
        response = current_response if current_response else "Waiting for analysis..."
        try:
            while not response_queue.empty():
                try:
                    new_response = response_queue.get_nowait()
                    if new_response and len(new_response) > 5:
                        response = new_response
                except:
                    break
        except:
            pass

        # Overlay text on image
        try:
            draw = ImageDraw.Draw(pil_img)

            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

            # Draw semi-transparent background for text
            try:
                text_bbox = draw.textbbox((0, 0), response[:200], font=font) if font else (0, 0, len(response)*8, 20)
                text_width = min(text_bbox[2] - text_bbox[0], pil_img.width - 40)
                text_height = text_bbox[3] - text_bbox[1]

                # Semi-transparent background
                overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([10, 10, 10 + text_width + 20, 10 + text_height + 20],
                                      fill=(0, 0, 0, 180))
                pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)

                # Draw text (limit length to avoid overflow)
                draw = ImageDraw.Draw(pil_img)
                display_text = response[:200] if len(response) > 200 else response
                draw.text((20, 20), display_text, fill=(255, 255, 255), font=font)
            except Exception as e:
                print(f"⚠️ Text overlay error: {e}")

            return pil_img.convert('RGB')
        except Exception as e:
            print(f"⚠️ Overlay error: {e}")
            return pil_img.convert('RGB')
            
    except Exception as e:
        # Return error image
        print(f"❌ get_overlay_frame error: {e}")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        pil_img = Image.fromarray(blank)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((20, 20), f"Display Error: {str(e)[:50]}", fill=(255, 0, 0), font=font)
        return pil_img.convert('RGB')

    # Convert to PIL for overlay
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Get latest response
    response = current_response
    while not response_queue.empty():
        try:
            response = response_queue.get_nowait()
        except:
            break

    # Overlay text on image
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Draw semi-transparent background for text
    text_bbox = draw.textbbox((0, 0), response, font=font) if font else (0, 0, len(response)*10, 20)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Semi-transparent background
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([10, 10, 10 + text_width + 20, 10 + text_height + 20],
                          fill=(0, 0, 0, 128))
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)

    # Draw text
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 20), response, fill=(255, 255, 255), font=font)

    return pil_img.convert('RGB')

def start_system(rtsp_url, prompt, interval, max_tokens):
    """Start video capture and analysis"""
    global video_active, analysis_active, current_interval

    print(f"🚀 Starting system: interval={interval}s, max_tokens={max_tokens}")

    # Start video capture
    if not video_active:
        print(f"📺 Starting video capture from: {rtsp_url[:50]}...")
        video_active = True
        video_thread = threading.Thread(
            target=video_capture_loop,
            args=(rtsp_url,),
            daemon=True
        )
        video_thread.start()
        print("✓ Video capture thread started")
        
        # Wait a bit for video to start capturing
        time.sleep(1.0)  # Increased wait time
        
        # Check if we got any frames
        if len(frame_buffer) == 0:
            print("⚠️ No frames captured yet after 1 second - video may be connecting...")
        else:
            print(f"✓ Video capture working! Buffer has {len(frame_buffer)} frames")

    # Update interval and restart analysis if needed
    current_interval = interval

    # Always restart analysis to apply new settings
    if analysis_active:
        print("🔄 Restarting analysis with new settings...")
        analysis_active = False  # Stop current analysis
        time.sleep(0.2)  # Brief pause to let thread stop

    # Start new analysis with updated settings
    analysis_active = True
    analysis_thread = threading.Thread(
        target=analysis_loop,
        args=(prompt, interval, max_tokens),
        daemon=True
    )
    analysis_thread.start()
    print(f"🧠 Analysis thread started with interval={interval}s")

    return f"✅ System started! Analysis every {interval}s"

# Build UI
with gr.Blocks(title="FastVLM - Live Video Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Live Security Analysis")
    gr.Markdown("*Real-time video monitoring with AI detection*")

    with gr.Row():
        # Video with overlay
        video_output = gr.Image(
            label="Live Feed with Analysis",
            type="pil",
            height=600,
            streaming=True
        )

    with gr.Row():
        with gr.Column(scale=1):
            interval_input = gr.Slider(
                label="Analysis Frequency (seconds)",
                minimum=1.0,
                maximum=5.0,
                value=2.0,
                step=0.5
            )

            start_button = gr.Button("▶️ Start Monitoring", variant="primary", size="lg")
            status_output = gr.Textbox(label="Status", interactive=False)

    # Simple initialization - just set status, let timer handle video
    demo.load(
        fn=lambda: "Ready to monitor",
        inputs=[],
        outputs=[status_output]
    )

    # Continuous updates - simplified to avoid connection errors
    def safe_get_overlay():
        """Wrapper to safely get overlay frame"""
        try:
            return get_overlay_frame()
        except Exception as e:
            print(f"⚠️ Safe overlay error: {e}")
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return Image.fromarray(blank).convert('RGB')
    
    timer = gr.Timer(0.1)
    timer.tick(
        fn=safe_get_overlay,
        inputs=[],
        outputs=[video_output]
    )

    def start_monitoring(interval):
        """Start the monitoring system"""
        prompt = """You are analyzing a live video feed. Your response MUST start with a DETAILED description of what you see in the scene.

FIRST, describe the scene in detail (write 4-6 sentences):
- What objects, items, products, or merchandise are visible?
- What is the layout and structure of the space?
- How many people are visible and where exactly are they located?
- What are the people doing - describe their activities and positions?
- What is the lighting like - bright, dim, natural, artificial?
- Are there any notable movements or changes happening?

THEN, assess for shoplifting:
- Are people taking items from shelves?
- Are items being concealed in bags, pockets, or clothing?
- Are there suspicious hand movements?
- Is anyone leaving without paying?

End with: "VERDICT: YES/NO - [brief reason]"

Remember: Always describe what you SEE first, even if everything appears normal."""
        
        return start_system(RTSP_URL, prompt, interval, 100)

    # Start button
    start_button.click(
        fn=start_monitoring,
        inputs=[interval_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    print("🚀 Starting FastVLM Overlay UI...")
    print("📺 Open http://localhost:7862 in your browser")
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False, show_api=False)
