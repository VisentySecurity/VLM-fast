#!/usr/bin/env python3
"""
FastVLM RTSP Stream Processor
Real-time video analysis using FastVLM model from HuggingFace
"""

import cv2
import torch
import time
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# Configuration
RTSP_URL = "rtsp://11QgE4OL:qrIW0btYlBUouIDZ@10.104.8.173:1000/live/ch0"
MODEL_NAME = "apple/FastVLM-0.5B"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROMPT = "Answer: Yes or No. Is shoplifting happening? Then describe what you see in 1-2 sentences."
FRAME_INTERVAL = 1.0  # Process every 1 second
MAX_TOKENS = 60  # Enough for Yes/No + description

print(f"🚀 FastVLM RTSP Stream Processor")
print(f"📹 RTSP URL: {RTSP_URL}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"💻 Device: {DEVICE}")
print(f"⏱️  Frame interval: {FRAME_INTERVAL}s")
print(f"🔢 Max tokens: {MAX_TOKENS}")
print("-" * 60)

# Load model
print("📥 Loading model from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE.type == "mps" else torch.float32,
    device_map={"": DEVICE.type},
    trust_remote_code=True,
)
model.eval()

# Get image processor
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
image_processor = vision_tower.image_processor

print("✅ Model loaded successfully!")
print("-" * 60)

# Open RTSP stream
print(f"🔌 Connecting to RTSP stream...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print(f"❌ Error: Could not open RTSP stream: {RTSP_URL}")
    exit(1)

print("✅ Connected to RTSP stream!")
print("-" * 60)
print("🎬 Starting real-time analysis...")
print("📺 Video window will open - Press 'q' to quit or Ctrl+C to stop")
print("-" * 60)

# Create display window
cv2.namedWindow("FastVLM - Shoplifting Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FastVLM - Shoplifting Detection", 1280, 720)

frame_count = 0
last_process_time = 0
current_response = "Analyzing..."
current_status = "✅ NO"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read frame, retrying...")
            time.sleep(1)
            continue
        
        current_time = time.time()
        
        # Always display the video frame
        display_frame = frame.copy()
        
        # Add text overlay on video
        status_color = (0, 0, 255) if "YES" in current_status else (0, 255, 0)  # Red for YES, Green for NO
        cv2.putText(display_frame, current_status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Add description (wrap text if needed)
        description_lines = []
        words = current_response.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) < 60:
                current_line = test_line
            else:
                if current_line:
                    description_lines.append(current_line)
                current_line = word
        if current_line:
            description_lines.append(current_line)
        
        y_offset = 70
        for line in description_lines[:3]:  # Show max 3 lines
            cv2.putText(display_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Show frame count and FPS
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow("FastVLM - Shoplifting Detection", display_frame)
        
        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Process frame at specified interval
        if current_time - last_process_time >= FRAME_INTERVAL:
            frame_count += 1
            last_process_time = current_time
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Build prompt
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT}"
            messages = [
                {"role": "user", "content": qs}
            ]
            
            # Use tokenizer's chat template
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
            start_time = time.time()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half(),
                    image_sizes=[pil_image.size],
                    max_new_tokens=MAX_TOKENS,
                    min_new_tokens=15,  # Minimum for Yes/No + brief description
                    do_sample=False,  # Greedy decoding for speed
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Extract response - try multiple methods to get full response
            input_token_len = input_ids.shape[1]
            generated_ids = output_ids[:, input_token_len:]
            
            # Method 1: Decode only generated tokens
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Method 2: If response seems incomplete, decode full output and extract
            if not response or len(response) < 5 or response.startswith(','):
                full_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                prompt_decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0].strip()
                
                # Try to extract response after prompt
                if full_output.startswith(prompt_decoded):
                    response = full_output[len(prompt_decoded):].strip()
                else:
                    # Look for common response patterns
                    for marker in ["Answer:", "Yes", "No", "The", "This", "I see"]:
                        if marker in full_output:
                            idx = full_output.find(marker)
                            if idx > len(prompt_decoded) // 2:  # Response should be after prompt
                                response = full_output[idx:].strip()
                                break
                    if not response or response.startswith(','):
                        response = full_output  # Fallback to full output
            
            # Clean up response - remove leading commas, whitespace
            response = response.strip()
            while response and response[0] in (',', '.', ' ', '\n', '\t'):
                response = response[1:].strip()
            
            # Debug: Check if response is still empty
            if not response or len(response) < 3:
                print(f"⚠️  Warning: Empty or very short response. Generated tokens: {generated_ids.shape[1]}")
                response = "No description available"
            
            # Parse response for shoplifting detection
            response_lower = response.lower()
            is_shoplifting = False
            if "yes" in response_lower:
                # Check if Yes is related to shoplifting
                if any(word in response_lower for word in ["shoplift", "steal", "theft", "taking", "removing"]):
                    is_shoplifting = True
                # Also check if Yes appears early in response (likely answer to question)
                elif response_lower.find("yes") < 10:
                    is_shoplifting = True
            
            inference_time = time.time() - start_time
            
            # Store current response for display
            current_response = response
            current_status = "🚨 YES - SHOPLIFTING DETECTED" if is_shoplifting else "✅ NO - Normal activity"
            
            # Display results in console
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Frame #{frame_count}")
            print(f"⏱️  Inference: {inference_time:.2f}s")
            print(f"🔍 Shoplifting: {current_status}")
            print(f"📝 Description: {current_response}")
            print("-" * 60)
        
        # Small delay to prevent CPU spinning
        time.sleep(0.03)  # Reduced for smoother video display

except KeyboardInterrupt:
    print("\n\n🛑 Stopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Stream closed. Goodbye!")

