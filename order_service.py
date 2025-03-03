from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uvicorn
import cv2
import torch
import numpy as np
import time
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import signal

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store orders and active WebSocket connections
orders = []
active_connections = []

# Store current inventory and last order times
inventory = {}
last_order_times = {}
ORDER_COOLDOWN = 1200  # 20 minutes in seconds

# Load YOLO Model
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress future warnings

# Load YOLO Model with CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
model.conf = 0.5  # Set confidence threshold for detection
model.classes = [39, 73, 41]  # Class indices for bottle (39), cup (41), and box-like objects (73)

# Dictionary to map COCO class indices to our inventory items
class_mapping = {
    39: 'bottle',
    73: 'box',
    41: 'cup'
}

# Initialize camera with retry mechanism
def init_camera():
    try:
        # Try DirectShow first
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Set lower resolution to improve performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Successfully initialized camera with DirectShow")
                return cap
            cap.release()
    except Exception as e:
        print(f"DirectShow initialization failed: {e}")

    try:
        # Try default backend
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Set lower resolution to improve performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Successfully initialized camera with default backend")
                return cap
            cap.release()
    except Exception as e:
        print(f"Default backend initialization failed: {e}")

    return None

# Video capture for streaming with retry mechanism
camera = None
max_retries = 3
retry_count = 0

print("🎥 Initializing camera...")
while camera is None and retry_count < max_retries:
    try:
        camera = init_camera()
        if camera is None:
            print(f"Camera initialization attempt {retry_count + 1} of {max_retries} failed")
            retry_count += 1
            time.sleep(2)
    except Exception as e:
        print(f"Error during camera initialization: {str(e)}")
        retry_count += 1
        time.sleep(2)

if camera is None:
    print("Warning: Could not initialize camera. Starting without camera support.")
else:
    print("Camera initialized successfully!")

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "patelneel161004@gmail.com"  # Your email
SENDER_PASSWORD = "lbud suty psdi urmo"  # Your Gmail App Password
SUPPLIER_EMAIL = "neel_patel2004@outlook.com"  # Supplier email

# Notification thresholds and order quantities
NOTIFICATION_THRESHOLDS = {
    'bottle': 3,
    'box': 3,
    'cup': 3
}

# Auto-order quantities
AUTO_ORDER_QUANTITIES = {
    'bottle': 10,
    'box':10,
    'cup': 10
}

# Track last notification time and order times
last_notification_time = {}
last_order_times = {}
NOTIFICATION_COOLDOWN = 1200  # 20 minutes in seconds
ORDER_COOLDOWN = 1200  # 20 minutes in seconds

def send_notification(item_name, current_stock):
    current_time = time.time()
    
    # Check cooldown period
    if item_name in last_notification_time:
        if current_time - last_notification_time[item_name] < NOTIFICATION_COOLDOWN:
            return False

    subject = f"Low Stock Alert: {item_name}"
    body = f"The stock for {item_name} is running low. Current count: {current_stock}. Please restock soon."

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = SUPPLIER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, SUPPLIER_EMAIL, msg.as_string())
        server.quit()
        print(f"Notification sent for {item_name}")
        last_notification_time[item_name] = current_time
        return True
    except Exception as e:
        print(f"Failed to send notification: {e}")
        return False

async def check_and_place_order(item_name, current_count):
    current_time = time.time()
    
    # Check if we have at least one item detected
    if current_count >= 1:
        # Check if enough time has passed since last order
        if item_name not in last_order_times or (current_time - last_order_times[item_name]) >= ORDER_COOLDOWN:
            try:
                # Place the order
                order_request = OrderRequest(
                    item_name=item_name,
                    quantity=AUTO_ORDER_QUANTITIES.get(item_name, 1),
                    order_type=OrderType.AUTOMATIC
                )
                
                # Create and process the order
                order = Order(
                    item_name=order_request.item_name,
                    quantity=order_request.quantity,
                    timestamp=datetime.now(),
                    order_id=len(orders) + 1,
                    order_type=order_request.order_type
                )
                orders.append(order)
                
                # Update last order time
                last_order_times[item_name] = current_time
                
                # Update order status and notify clients
                order.status = OrderStatus.PROCESSING
                await notify_clients(order)
                
                print(f"Automatic order placed for {item_name}")
                
                # Complete order and notify clients again
                order.status = OrderStatus.COMPLETED
                await notify_clients(order)
                
                # Send notification email
                send_notification(item_name, current_count)
                
                return True
            except Exception as e:
                print(f"Failed to place automatic order for {item_name}: {e}")
                return False
    return False

async def generate_frames():
    global camera, inventory  # Add inventory to global variables
    while True:
        try:
            if camera is None:
                print("Camera not available, attempting to initialize...")
                camera = init_camera()
                if camera is None:
                    await asyncio.sleep(2)
                    continue

            success, frame = camera.read()
            if not success:
                print("Failed to grab frame, reinitializing camera...")
                camera.release()
                camera = init_camera()
                if camera is None:
                    await asyncio.sleep(2)
                    continue
                continue

            # Perform object detection
            results = model(frame)
            detections = results.pandas().xyxy[0]
            
            # Count items by category
            item_counts = {}
            for _, row in detections.iterrows():
                class_idx = int(row['class'])
                if class_idx in class_mapping:
                    item_name = class_mapping[class_idx]
                    item_counts[item_name] = item_counts.get(item_name, 0) + 1
            
            # Update inventory counts and check thresholds
            current_time = datetime.now()
            for item_name, count in item_counts.items():
                # Update inventory dictionary
                inventory[item_name] = {
                    "count": count,
                    "timestamp": current_time
                }
                
                # Check if we should place an order
                await check_and_place_order(item_name, count)
                
                # Prepare update message
                update_message = {
                    "type": "inventory_update",
                    "data": {
                        "item_name": item_name,
                        "count": count,
                        "timestamp": current_time.isoformat()
                    }
                }
                
                # Notify clients about inventory update
                for connection in active_connections[:]:
                    try:
                        await connection.send_json(update_message)
                    except Exception as e:
                        print(f"WebSocket error: {e}")
                        if connection in active_connections:
                            active_connections.remove(connection)
            
            # Draw bounding boxes
            for _, row in detections.iterrows():
                class_idx = int(row['class'])
                if class_idx in class_mapping:
                    item_name = class_mapping[class_idx]
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{item_name} ({row['confidence']:.2f})", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            await asyncio.sleep(2)  # Wait before retrying

class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

class OrderType(str, Enum):
    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

class Order(BaseModel):
    item_name: str
    quantity: int
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: int = None
    order_type: OrderType = OrderType.MANUAL

    class Config:
        use_enum_values = True

class InventoryUpdate(BaseModel):
    item_name: str
    count: int
    timestamp: datetime

class OrderRequest(BaseModel):
    item_name: str
    quantity: int
    order_type: OrderType = OrderType.AUTOMATIC

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)

async def notify_clients(order: Order):
    message = {
        "type": "order_update",  
        "data": {
            "order_id": order.order_id,
            "item_name": order.item_name,
            "quantity": order.quantity,
            "timestamp": order.timestamp.isoformat(),
            "status": order.status,
            "order_type": order.order_type
        }
    }
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            active_connections.remove(connection)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/place_order")
async def place_order(order_request: OrderRequest):
    current_time = time.time()
    
    # Check if enough time has passed since last order
    if order_request.item_name in last_order_times:
        time_since_last_order = current_time - last_order_times[order_request.item_name]
        if time_since_last_order < ORDER_COOLDOWN:
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {int((ORDER_COOLDOWN - time_since_last_order) / 60)} minutes before placing another order for {order_request.item_name}"
            )
    
    order = Order(
        item_name=order_request.item_name,
        quantity=order_request.quantity,
        timestamp=datetime.now(),
        order_id=len(orders) + 1,
        order_type=order_request.order_type
    )
    orders.append(order)
    
    # Update last order time
    last_order_times[order_request.item_name] = current_time
    
    # Simulate order processing
    order.status = OrderStatus.PROCESSING
    await notify_clients(order)
    
    # In a real system, you'd process the order here
    order.status = OrderStatus.COMPLETED
    await notify_clients(order)
    
    return {
        "status": "success",
        "message": f"Order placed for {order_request.quantity} units of {order_request.item_name}",
        "order_id": order.order_id,
        "timestamp": order.timestamp,
        "order_type": order_request.order_type,
        "next_available_order": datetime.fromtimestamp(current_time + ORDER_COOLDOWN).isoformat()
    }

@app.get("/orders")
async def get_orders():
    return orders

@app.post("/update_inventory")
async def update_inventory(update: InventoryUpdate):
    inventory[update.item_name] = {
        "count": update.count,
        "timestamp": update.timestamp
    }
    # Notify all clients about inventory update
    for connection in active_connections:
        try:
            await connection.send_json({
                "type": "inventory_update",
                "data": {
                    "item_name": update.item_name,
                    "count": update.count,
                    "timestamp": update.timestamp.isoformat()
                }
            })
        except:
            active_connections.remove(connection)
    return {"status": "success"}

@app.get("/inventory")
async def get_inventory():
    return inventory

@app.get("/video_feed")
async def video_feed():
    if camera is None:
        # Return a 503 Service Unavailable status if camera is not available
        raise HTTPException(
            status_code=503,
            detail="Camera not available"
        )
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    if camera is not None:
        camera.release()
    print("✅ Camera released")
    import sys
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        server.run()
    finally:
        if camera is not None:
            camera.release()
        print("Camera released") 