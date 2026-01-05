#!/usr/bin/env python3
"""
FastVLM Real-time Video Overlay
Clean video feed with AI analysis overlaid on screen
"""

import cv2
import torch
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# Configuration
RTSP_URL = "rtsp://11QgE4OL:qrIW0btYlBUouIDZ@10.104.8.173:1000/live/ch0"
MODEL_NAME = "apple/FastVLM-0.5B"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROMPT = "Is someone taking items from a table and putting them into a bag, coat, or personal item? Answer: YES or NO, then briefly describe what you see."
FRAME_INTERVAL = 1.0  # Process every 1 second

# Global model cache
_model_cache = {}

def load_model():
    """Load model once and cache it"""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"], _model_cache["image_processor"]
    
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
    
    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    _model_cache["image_processor"] = image_processor
    
    print("✅ Model loaded!")
    return model, tokenizer, image_processor

def analyze_frame(frame, model, tokenizer, image_processor):
    """Analyze a single frame"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Build prompt
    qs = f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT}"
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
    
    # Generate - fast and focused
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[pil_image.size],
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract response
    input_token_len = input_ids.shape[1]
    generated_ids = output_ids[:, input_token_len:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return response

def draw_overlay(frame, text, is_alert=False):
    """Draw text overlay on video frame"""
    # Convert to PIL for text rendering
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Background color for text
    bg_color = (255, 0, 0, 180) if is_alert else (0, 0, 0, 180)
    text_color = (255, 255, 255)
    
    # Split text into lines if too long
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] < frame.shape[1] - 40:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    # Draw background and text
    y_offset = 20
    for line in lines[:3]:  # Max 3 lines
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw semi-transparent background
        overlay = pil_image.copy()
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [(10, y_offset), (20 + text_width, y_offset + text_height + 10)],
            fill=bg_color[:3] + (180,)
        )
        pil_image = Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(pil_image)
        
        # Draw text
        draw.text((15, y_offset + 5), line, fill=text_color, font=font)
        y_offset += text_height + 15
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    """Main video overlay loop"""
    print("🚀 FastVLM Real-time Video Overlay")
    print(f"📹 RTSP: {RTSP_URL}")
    print(f"🤖 Model: {MODEL_NAME}")
    print("-" * 60)
    
    # Load model
    model, tokenizer, image_processor = load_model()
    
    # Open RTSP stream
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"❌ Error: Could not open RTSP stream")
        return
    
    print("✅ Connected to video stream")
    print("Press 'q' to quit")
    print("-" * 60)
    
    frame_count = 0
    last_analysis_time = 0
    current_analysis = "Analyzing..."
    is_alert = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Analyze frame at intervals
            if current_time - last_analysis_time >= FRAME_INTERVAL:
                try:
                    analysis = analyze_frame(frame, model, tokenizer, image_processor)
                    current_analysis = analysis
                    
                    # Check for alerts
                    analysis_lower = analysis.lower()
                    shoplifting_keywords = ["yes", "taking", "bag", "coat", "pocket", "stealing", "removing", "placing", "putting"]
                    is_alert = any(keyword in analysis_lower for keyword in shoplifting_keywords) and "no" not in analysis_lower[:10]
                    
                    if is_alert:
                        current_analysis = f"🚨 ALERT: {analysis}"
                    
                    last_analysis_time = current_time
                except Exception as e:
                    current_analysis = f"Error: {str(e)}"
                    is_alert = False
            
            # Draw overlay on frame
            frame_with_overlay = draw_overlay(frame, current_analysis, is_alert)
            
            # Display frame
            cv2.imshow("FastVLM Real-time Analysis", frame_with_overlay)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Closed")

if __name__ == "__main__":
    main()

