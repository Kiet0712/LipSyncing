# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import asyncio
import json
import base64
import logging
import sys
import os
import torch
import uuid # For generating unique session IDs
import redis # Import redis client
import pika # Import pika for RabbitMQ

# Configure logging for main.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory of SadTalker to sys.path to allow absolute imports like SadTalker.inference
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming SadTalker is a sibling directory to where main.py resides
sadtalker_path = os.path.join(current_dir, 'SadTalker')
if sadtalker_path not in sys.path:
    sys.path.append(sadtalker_path)
    logger.info(f"Added {sadtalker_path} to sys.path")

try:
    # We still import initialize_sadtalker_models for potential global initialization
    # but run_lipsync_inference_from_bytes will primarily be used by the worker.
    from inference import initialize_sadtalker_models
    from inference import DEFAULT_CHECKPOINT_DIR, DEFAULT_CONFIG_DIR
    logger.info("Successfully imported SadTalker inference components (for initialization).\n"
                "Note: Actual inference runs on ai_worker.py.")
except ImportError as e:
    logger.error(f"Failed to import SadTalker inference components: {e}")
    logger.error("Please ensure the 'SadTalker' directory is structured correctly and all dependencies are installed.")
    sys.exit(1)

app = FastAPI(
    title="Real-time LipSyncing AI WebSocket API (Queue Manager)",
    description="Real-time LipSyncing with image and audio input via WebSockets, using RabbitMQ and Redis.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Redis connection for general use (session state, image storage, Pub/Sub)
try:
    redis_client = redis.Redis(host='redis', port=6379, db=0)
    redis_client.ping() # Test connection
    logger.info("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"FATAL ERROR: Could not connect to Redis: {e}. Please ensure Redis server is running.")
    sys.exit(1)

# RabbitMQ connection variables
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_QUEUE_NAME = "lipsync_jobs_queue" # Define a queue name for RabbitMQ

# Global RabbitMQ connection and channel
rabbitmq_connection = None
rabbitmq_channel = None

# --- FastAPI Startup Event for Global Model Initialization (Optional for main.py) ---
# This is mainly for the worker, but keeping it here for consistency or if main.py ever needs direct inference.
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup event triggered: Initializing SadTalker models globally (if not already)...")
    
    global_inference_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Detected device for global SadTalker model initialization: {global_inference_device}")

    try:
        initialize_sadtalker_models(
            device=global_inference_device,
            checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
            config_dir=DEFAULT_CONFIG_DIR,
            size=256,
            old_version=False,
            preprocess_type='crop'
        )
        logger.info("SadTalker models initialized globally successfully in main.py.")
    except Exception as e:
        logger.error(f"WARNING: Failed to initialize SadTalker models at startup in main.py: {e}", exc_info=True)
        # We don't sys.exit(1) here as main.py is a manager, not an inference worker.
        # It can still function without local inference models if the worker is healthy.

    # Initialize RabbitMQ connection
    global rabbitmq_connection, rabbitmq_channel
    try:
        connection_params = pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
        rabbitmq_connection = pika.BlockingConnection(connection_params)
        rabbitmq_channel = rabbitmq_connection.channel()
        rabbitmq_channel.queue_declare(queue=RABBITMQ_QUEUE_NAME, durable=True) # Durable queue for persistence
        logger.info(f"Successfully connected to RabbitMQ and declared queue: {RABBITMQ_QUEUE_NAME}")
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"FATAL ERROR: Could not connect to RabbitMQ: {e}. Please ensure RabbitMQ server is running.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL ERROR: Unexpected error during RabbitMQ initialization: {e}", exc_info=True)
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown event triggered: Closing RabbitMQ connection...")
    global rabbitmq_connection
    if rabbitmq_connection and not rabbitmq_connection.is_closed:
        try:
            rabbitmq_connection.close()
            logger.info("RabbitMQ connection closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection during shutdown: {e}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {} # Map session_id to WebSocket
        self.client_states: Dict[WebSocket, Dict[str, Any]] = {}
        self.pubsub_listener_task: asyncio.Task | None = None # To hold the asyncio task for Redis Pub/Sub

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        self.client_states[websocket] = {
            "state": "IDLE", # IDLE, AWAITING_IMAGE, AWAITING_AUDIO_STREAM_START, PROCESSING_AUDIO
            "session_id": session_id,
            "image_id": None, # Redis key for the image
            "audio_queue_key": f"audio_queue:{session_id}", # Redis key for audio chunks
            "audio_stream_ended_key": f"audio_stream_ended:{session_id}", # Redis key for audio stream end flag
            "enhancer_option": False, # Boolean: True for gfpgan, False for None
            "use_cpu": False,
            "current_audio_buffer": b"", # In-memory buffer for accumulating audio
            "segment_counter": 0, # Counter for video segments (audio chunks processed)
            "audio_processing_task": None # Task for _handle_audio_chunking_and_enqueue
        }
        logger.info(f"Client {websocket.client.host}:{websocket.client.port} connected with session ID: {session_id}")
        
        # Start the Pub/Sub listener if it's not already running or finished
        if self.pubsub_listener_task is None or self.pubsub_listener_task.done():
            self.pubsub_listener_task = asyncio.create_task(self._redis_pubsub_listener())
            logger.info("Started Redis Pub/Sub listener task.")

    def disconnect(self, websocket: WebSocket):
        session_id = None
        # Find the session_id associated with the disconnected websocket
        for sid, ws in list(self.active_connections.items()): # Use list() for safe iteration during deletion
            if ws == websocket:
                session_id = sid
                del self.active_connections[session_id]
                logger.info(f"Removed WebSocket for session ID: {session_id}")
                break

        client_state = self.client_states.pop(websocket, None)
        if client_state:
            # Clean up Redis keys associated with this session
            session_id = client_state["session_id"]
            image_id = client_state.get("image_id")
            audio_queue_key = client_state["audio_queue_key"]
            audio_stream_ended_key = client_state["audio_stream_ended_key"]

            if image_id and redis_client.exists(image_id):
                redis_client.delete(image_id)
                logger.info(f"Cleaned up Redis key {image_id} for session {session_id}")
            if redis_client.exists(audio_queue_key):
                redis_client.delete(audio_queue_key)
                logger.info(f"Cleaned up Redis key {audio_queue_key} for session {session_id}")
            if redis_client.exists(audio_stream_ended_key):
                redis_client.delete(audio_stream_ended_key)
                logger.info(f"Cleaned up Redis key {audio_stream_ended_key} for session {session_id}")
            
            # Cancel the audio processing task if it's still running for this specific client
            if client_state["audio_processing_task"] and not client_state["audio_processing_task"].done():
                client_state["audio_processing_task"].cancel()
                logger.info(f"Cancelled audio processing task for session {client_state['session_id']}")

            logger.info(f"Client {websocket.client.host}:{websocket.client.port} disconnected and state cleaned up.")
        else:
            logger.warning(f"Disconnected WebSocket not found in client_states: {websocket.client.host}:{websocket.client.port}")


    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except RuntimeError as e:
            logger.warning(f"Could not send text message to client {websocket.client.host}:{websocket.client.port}, connection likely closed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending text message to client {websocket.client.host}:{websocket.client.port}: {e}", exc_info=True)


    async def send_bytes_message(self, message: bytes, websocket: WebSocket):
        try:
            await websocket.send_bytes(message)
        except RuntimeError as e:
            logger.warning(f"Could not send bytes message to client {websocket.client.host}:{websocket.client.port}, connection likely closed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending bytes message to client {websocket.client.host}:{websocket.client.port}: {e}", exc_info=True)


    async def _redis_pubsub_listener(self):
        """
        Listens to Redis Pub/Sub channels for video segments from the AI worker.
        """
        pubsub = redis_client.pubsub()
        # Subscribe to a pattern that matches all session-specific result channels
        pubsub.psubscribe("lipsync_results:*") 
        logger.info("Redis Pub/Sub listener started, subscribed to 'lipsync_results:*'")

        try:
            while True:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1) # Timeout to allow task cancellation
                if message and message['type'] == 'pmessage':
                    channel = message['channel'].decode('utf-8')
                    data = message['data']
                    
                    try:
                        session_id = channel.split(':')[-1]
                        logger.debug(f"Received message on channel {channel} for session {session_id}")
                    except IndexError:
                        logger.error(f"Malformed Pub/Sub channel name: {channel}. Message data: {data[:50]}...")
                        continue

                    websocket = self.active_connections.get(session_id)
                    if websocket:
                        try:
                            result = json.loads(data.decode('utf-8'))
                            
                            if result.get("is_error"):
                                error_msg = result.get("error", "Unknown error from worker.")
                                logger.error(f"Received error from worker for session {session_id}: {error_msg}")
                                await self.send_personal_message(f"Server Error during video generation: {error_msg}", websocket)
                            else:
                                # Send the full JSON message to the client, including segment_type and stream_type
                                # The client will parse this JSON to get base64_content and segment metadata
                                await self.send_personal_message(json.dumps(result), websocket)
                                logger.info(f"Sent {result.get('segment_type', 'unknown')} {result.get('stream_type', 'unknown')} segment "
                                            f"for session {session_id}, audio segment {result.get('audio_input_segment_number')}.")

                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON from Pub/Sub message for session {session_id}: {data.decode('utf-8')[:100]}...", exc_info=True)
                            await self.send_personal_message("Internal server error: malformed data from worker.", websocket)
                        except Exception as e:
                            logger.error(f"Error processing and sending video segment to client {session_id}: {e}", exc_info=True)
                            await self.send_personal_message(f"Internal server error: failed to process video data. {str(e)[:50]}...", websocket)
                    else:
                        logger.warning(f"No active WebSocket for session ID {session_id}. Discarding video segment.")
                await asyncio.sleep(0.01) # Prevent busy-waiting
        except asyncio.CancelledError:
            logger.info("Redis Pub/Sub listener task cancelled.")
        except Exception as e:
            logger.error(f"Error in Redis Pub/Sub listener: {e}", exc_info=True)
        finally:
            pubsub.punsubscribe("lipsync_results:*") # Unsubscribe from all patterns
            pubsub.close() # Close the connection
            logger.info("Redis Pub/Sub listener unsubscribed and connection closed.")


manager = ConnectionManager()

@app.get("/")
async def get_index():
    return HTMLResponse(content=open("static/index.html").read(), status_code=200)

def _enqueue_lipsync_job(websocket: WebSocket, client_state: Dict[str, Any], audio_segment_bytes: bytes, segment_number: int, is_final_chunk: bool):
    """
    Enqueues a lipsync job to RabbitMQ.
    This function is now called from a background task in _handle_audio_chunking_and_enqueue.
    """
    try:
        # Ensure RabbitMQ connection and channel are available
        if rabbitmq_connection is None or rabbitmq_connection.is_closed or rabbitmq_channel is None:
            logger.error("RabbitMQ connection or channel not available. Cannot enqueue job.")
            raise RuntimeError("RabbitMQ not connected.")

        job_payload = {
            "session_id": client_state["session_id"],
            "image_id": client_state["image_id"],
            "audio_segment_base64": base64.b64encode(audio_segment_bytes).decode('utf-8'),
            "segment_number": segment_number,
            "is_final_chunk": is_final_chunk,
            "enhancer": client_state["enhancer_option"],
            "use_cpu": client_state["use_cpu"]
        }
        rabbitmq_channel.basic_publish(
            exchange='',
            routing_key=RABBITMQ_QUEUE_NAME,
            body=json.dumps(job_payload),
            properties=pika.BasicProperties(
                delivery_mode=2, # Make message persistent
            )
        )
        logger.info(f"Session {client_state['session_id']}: Enqueued audio segment {segment_number} (final: {is_final_chunk}).")
        asyncio.create_task(manager.send_personal_message(f"Enqueued audio segment {segment_number}.", websocket))

    except Exception as e:
        logger.error(f"Failed to enqueue job for session {client_state['session_id']}, segment {segment_number}: {e}", exc_info=True)
        asyncio.create_task(manager.send_personal_message(f"Server Error: Failed to enqueue audio segment. {str(e)}", websocket))
        # Set state to error to stop further processing for this session
        client_state["state"] = "ERROR" # Or some specific error state
        # Attempt to end the audio stream in Redis
        redis_client.set(client_state["audio_stream_ended_key"], "true")
        client_state["current_audio_buffer"] = b""
        client_state["audio_processing_task"] = None # Clear the task reference


async def _handle_audio_chunking_and_enqueue(websocket: WebSocket, client_state: Dict[str, Any]):
    """
    Background task to accumulate audio chunks from Redis and enqueue jobs.
    """
    session_id = client_state["session_id"]
    audio_queue_key = client_state["audio_queue_key"]
    audio_stream_ended_key = client_state["audio_stream_ended_key"]
    AUDIO_SEGMENT_SIZE_BYTES = 4000

    logger.info(f"Starting audio chunking and enqueue task for session {session_id}")
    try:
        while True:
            # BLPOP waits for data or timeout
            popped_data = redis_client.blpop(audio_queue_key, timeout=0.1)
            is_stream_ended = redis_client.get(audio_stream_ended_key) == b"true"

            if popped_data:
                _, chunk_data = popped_data
                client_state["current_audio_buffer"] += chunk_data
                logger.debug(f"Session {session_id}: Added chunk. Buffer size: {len(client_state['current_audio_buffer'])} bytes.")

            # Process a segment if buffer is full enough OR if stream ended and there's any data left
            if len(client_state["current_audio_buffer"]) >= AUDIO_SEGMENT_SIZE_BYTES or \
               (is_stream_ended and len(client_state["current_audio_buffer"]) > 0):
                
                is_final_chunk_for_job = False
                segment_to_process = b""

                if is_stream_ended and len(client_state["current_audio_buffer"]) > 0:
                    # If stream ended and there's any data left, process it as the final chunk
                    segment_to_process = client_state["current_audio_buffer"]
                    client_state["current_audio_buffer"] = b""
                    is_final_chunk_for_job = True
                    logger.info(f"Session {session_id}: Processing final audio chunk of size {len(segment_to_process)}.")

                    # Pad the final segment if it's too short for SadTalker
                    if len(segment_to_process) < AUDIO_SEGMENT_SIZE_BYTES:
                        padding_needed = AUDIO_SEGMENT_SIZE_BYTES - len(segment_to_process)
                        # Append zeros (silent PCM data) to pad the segment
                        segment_to_process += b'\x00' * padding_needed
                        logger.info(f"Session {session_id}: Padded final audio chunk with {padding_needed} bytes of silence.")

                elif len(client_state["current_audio_buffer"]) >= AUDIO_SEGMENT_SIZE_BYTES:
                    # Process a full segment
                    segment_to_process = client_state["current_audio_buffer"][:AUDIO_SEGMENT_SIZE_BYTES]
                    client_state["current_audio_buffer"] = client_state["current_audio_buffer"][AUDIO_SEGMENT_SIZE_BYTES:]
                    logger.info(f"Session {session_id}: Processing regular audio chunk of size {len(segment_to_process)}.")

                if segment_to_process:
                    client_state["segment_counter"] += 1
                    _enqueue_lipsync_job(
                        websocket,
                        client_state,
                        segment_to_process,
                        client_state["segment_counter"],
                        is_final_chunk_for_job
                    )
                
                if is_final_chunk_for_job:
                    logger.info(f"Session {session_id}: Audio stream processing task finished as final chunk was processed.")
                    break # Exit the loop if final chunk handled

            elif is_stream_ended and len(client_state["current_audio_buffer"]) == 0:
                logger.info(f"Session {session_id}: Audio stream ended and buffer is empty. Task finishing.")
                break # All audio processed and stream ended

            await asyncio.sleep(0.05) # Small delay to prevent busy-waiting

    except asyncio.CancelledError:
        logger.info(f"Audio chunking and enqueue task for session {session_id} cancelled.")
    except Exception as e:
        logger.error(f"Error in audio chunking and enqueue task for session {session_id}: {e}", exc_info=True)
        await manager.send_personal_message(f"An unexpected server error occurred during audio processing: {e}", websocket)
        redis_client.set(audio_stream_ended_key, "true")
        client_state["current_audio_buffer"] = b""
    finally:
        logger.info(f"Audio chunking and enqueue task for session {session_id} finished or failed. Final state: {client_state['state']}")
        if client_state["state"] == "PROCESSING_AUDIO":
            client_state["state"] = "IDLE" # Reset state to allow new session


@app.websocket("/ws/lipsync")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    client_state = manager.client_states[websocket]
    client_state["state"] = "AWAITING_IMAGE" # Initial state for new client
    try:
        await manager.send_personal_message("Ready. Send 'image_init' with base64 content.", websocket)
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                data = message.get("text")
                if isinstance(data, str):
                    try:
                        parsed_data = json.loads(data)
                        if not isinstance(parsed_data, dict) or "type" not in parsed_data:
                            logger.warning(f"Session {client_state['session_id']}: Received JSON without 'type' field or not a dict: {data[:100]}...")
                            await manager.send_personal_message("Server Error: Received malformed JSON (missing 'type' or not a dict).", websocket)
                            continue
                        msg_type = parsed_data.get("type")
                        if not isinstance(msg_type, str):
                            logger.warning(f"Session {client_state['session_id']}: Received JSON with non-string 'type' field: {msg_type}. Raw: {data[:100]}...")
                            await manager.send_personal_message("Server Error: Received malformed JSON ('type' field is not a string).", websocket)
                            continue

                        if msg_type == "image_init" and client_state["state"] == "AWAITING_IMAGE":
                            base64_image_content = parsed_data.get("base64_content")
                            if not base64_image_content:
                                logger.warning(f"Session {client_state['session_id']}: Image metadata missing 'base64_content'.")
                                await manager.send_personal_message("Client Error: Image content missing. Please provide image.", websocket)
                                continue
                            
                            try:
                                image_bytes = base64.b64decode(base64_image_content)
                                image_id = f"image:{client_state['session_id']}"
                                redis_client.set(image_id, image_bytes)
                                redis_client.expire(image_id, 3600) # Expire image after 1 hour

                                client_state["image_id"] = image_id
                                client_state["enhancer_option"] = parsed_data.get("enhance_face", False)
                                client_state["use_cpu"] = parsed_data.get("use_cpu", False) # Get use_cpu setting
                                client_state["state"] = "AWAITING_AUDIO_STREAM_START"
                                await manager.send_personal_message("Image received. Ready for audio chunks.", websocket)
                                logger.info(f"Session {client_state['session_id']}: Image {image_id} stored in Redis. Enhance face: {client_state['enhancer_option']}, Use CPU: {client_state['use_cpu']}")

                            except Exception as e:
                                logger.error(f"Session {client_state['session_id']}: Error processing image: {e}", exc_info=True)
                                await manager.send_personal_message(f"Server Error processing image: {e}", websocket)
                                client_state["state"] = "IDLE" # Reset state on error

                        elif msg_type == "start_audio_stream" and client_state["state"] == "AWAITING_AUDIO_STREAM_START":
                            if not client_state["audio_processing_task"] or client_state["audio_processing_task"].done():
                                client_state["state"] = "PROCESSING_AUDIO"
                                client_state["segment_counter"] = 0 # Reset segment counter for a new stream
                                # Clear any old audio stream ended flag in Redis
                                redis_client.delete(client_state["audio_stream_ended_key"])
                                # Start the background task to handle audio chunking and enqueuing
                                client_state["audio_processing_task"] = asyncio.create_task(
                                    _handle_audio_chunking_and_enqueue(websocket, client_state)
                                )
                                logger.info(f"Session {client_state['session_id']}: Audio stream initiated. Starting processing task.")
                                await manager.send_personal_message("Audio stream initiated. Sending chunks now.", websocket)
                            else:
                                logger.warning(f"Session {client_state['session_id']}: Received start_audio_stream but processing task already running. Ignoring.")
                                await manager.send_personal_message("Audio stream already active. Please stop current recording first if starting a new one.", websocket)

                        elif msg_type == "audio_chunk" and client_state["image_id"] is not None and client_state["state"] in ["AWAITING_AUDIO_STREAM_START", "PROCESSING_AUDIO"]:
                            base64_audio_content = parsed_data.get("base64_content")
                            if not base64_audio_content:
                                logger.warning(f"Session {client_state['session_id']}: Audio chunk missing 'base64_content'.")
                                await manager.send_personal_message("Client Error: Audio chunk missing content. Skipping.", websocket)
                                continue
                            try:
                                audio_chunk_bytes = base64.b64decode(base64_audio_content)
                                if audio_chunk_bytes:
                                    redis_client.rpush(client_state["audio_queue_key"], audio_chunk_bytes)
                                    logger.debug(f"Session {client_state['session_id']}: Received audio chunk ({len(audio_chunk_bytes)} bytes). Redis queue len: {redis_client.llen(client_state['audio_queue_key'])}")
                                else:
                                    logger.warning(f"Session {client_state['session_id']}: Received empty audio chunk.")
                            except Exception as e:
                                logger.error(f"Session {client_state['session_id']}: Error decoding audio chunk: {e}", exc_info=True)
                                await manager.send_personal_message("Client Error: Could not decode audio data.", websocket)

                        elif msg_type == "end_audio_stream" and client_state["state"] == "PROCESSING_AUDIO":
                            logger.info(f"Session {client_state['session_id']}: Received end_audio_stream message.")
                            redis_client.set(client_state["audio_stream_ended_key"], "true")
                            await manager.send_personal_message("Audio stream ended. Finishing processing...", websocket)
                            client_state["state"] = "IDLE" 
                            logger.info(f"Session {client_state['session_id']}: State set to IDLE after end_audio_stream.")

                        else:
                            logger.warning(f"Session {client_state['session_id']}: Received unexpected message type '{msg_type}' in state '{client_state['state']}'.")
                            await manager.send_personal_message(f"Client Error: Unexpected message type '{msg_type}' in current state.", websocket)

                    except json.JSONDecodeError:
                        logger.warning(f"Session {client_state['session_id']}: Received non-JSON text data: {data[:100]}. Expected JSON text.")
                        await manager.send_personal_message("Client Error: Received unexpected non-JSON text data. Expected JSON text.", websocket)
                    except Exception as e:
                        logger.error(f"Session {client_state['session_id']}: Error processing JSON message: {e}", exc_info=True)
                        await manager.send_personal_message(f"Client Error: Error processing JSON message: {e}", websocket)

                elif isinstance(data, bytes):
                    logger.warning(f"Session {client_state['session_id']}: Received unexpected binary data. Expected JSON text.")
                    await manager.send_personal_message("Client Error: Received unexpected binary data. Expected JSON text.", websocket)

                else:
                    logger.warning(f"Session {client_state['session_id']}: Received unknown data type: {type(data)} in state: {client_state['state']}")
                    await manager.send_personal_message(f"Client Error: Received unknown data type. Expected JSON text.", websocket)

            else:
                logger.warning(f"Session {client_state['session_id']}: Received unexpected WebSocket message type: {message['type']}")
                await manager.send_personal_message(f"Client Error: Received unexpected WebSocket message type: {message['type']}.", websocket)

    except WebSocketDisconnect:
        logger.info(f"Client {websocket.client.host}:{websocket.client.port} disconnected cleanly.")
    except Exception as e:
        logger.error(f"WebSocket error for client {websocket.client.host}:{websocket.client.port}: {e}", exc_info=True)
        try:
            await manager.send_personal_message(f"An unexpected server error occurred: {e}", websocket)
        except RuntimeError:
            logger.warning("Could not send error message to client, connection already disconnected.")
    finally:
        manager.disconnect(websocket)
