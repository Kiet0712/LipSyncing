import asyncio
import logging
import os
import sys
import torch
import base64
import redis
import pika # Import pika for RabbitMQ
import signal
import json
import wave
import tempfile
import subprocess # Import subprocess for running ffmpeg
from glob import glob # For finding generated files
import re # For regex parsing of segment names
import shutil # For robust directory cleanup

def pcm_to_wav(pcm_bytes, wav_path, sample_rate=16000, sample_width=2, channels=1):
    """
    Converts raw PCM audio bytes to a WAV file.

    Args:
        pcm_bytes (bytes): The raw PCM audio data.
        wav_path (str): The path where the WAV file will be saved.
        sample_rate (int): The sample rate of the PCM data (e.g., 16000 Hz).
        sample_width (int): The width of each sample in bytes (e.g., 2 for 16-bit).
        channels (int): The number of audio channels (e.g., 1 for mono).
    """
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

# Configure logging for ai_worker.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory of SadTalker to sys.path to allow absolute imports like SadTalker.inference
current_dir = os.path.dirname(os.path.abspath(__file__))
sadtalker_path = os.path.join(current_dir, 'SadTalker')
if sadtalker_path not in sys.path:
    sys.path.append(sadtalker_path)
    logger.info(f"Added {sadtalker_path} to sys.path")

try:
    from inference import run_lipsync_inference_from_bytes, initialize_sadtalker_models
    from inference import DEFAULT_CHECKPOINT_DIR, DEFAULT_CONFIG_DIR
    logger.info("Successfully imported SadTalker inference components for worker.")
except ImportError as e:
    logger.error(f"Failed to import SadTalker inference components in worker: {e}")
    logger.error("Please ensure the 'SadTalker' directory is structured correctly and all dependencies are installed.")
    sys.exit(1)

# Initialize Redis client for worker (for fetching image and publishing results)
try:
    redis_client = redis.Redis(host='redis', port=6379, db=0)
    redis_client.ping() # Test connection
    logger.info("Worker successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"FATAL ERROR: Could not connect to Redis: {e}")
    sys.exit(1)

# --- NEW: Define Redis keys for tracking init segment status ---
REDIS_SESSION_INIT_STATUS_KEY = "session_init_status" # Redis Hash name
# Hash fields will be like: "session_id:video_init_sent", "session_id:audio_init_sent"


# RabbitMQ connection variables
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_QUEUE_NAME = "lipsync_jobs_queue" # Must match the queue name in main.py

# Global RabbitMQ connection and channel for the worker
rabbitmq_connection = None
rabbitmq_channel = None

# Global worker running flag to manage worker lifecycle
_worker_running = False

# Initialize SadTalker models once at the start of the worker process
# This ensures the worker is ready to process jobs as soon as it starts.
logger.info("Initializing SadTalker models for worker process...")
worker_inference_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Detected device for SadTalker model initialization in worker: {worker_inference_device}")
try:
    initialize_sadtalker_models(
        device=worker_inference_device,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        config_dir=DEFAULT_CONFIG_DIR,
        size=256,
        old_version=False,
        preprocess_type='crop'
    )
    logger.info("SadTalker models initialized successfully in ai_worker.py.")
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to initialize SadTalker models in worker: {e}", exc_info=True)
    sys.exit(1)

# NEW: Directory to save processed video segments
SAVE_VIDEO_DIR = os.getenv('SAVE_VIDEO_DIR', 'processed_videos')
# Ensure the directory exists
os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
logger.info(f"Videos will be saved to: {os.path.abspath(SAVE_VIDEO_DIR)}")


def _run_command(command, description="command"):
    """Helper to run shell commands and log their output."""
    logger.info(f"Executing {description}: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug(f"{description} stdout:\n{process.stdout}")
        # Only log stderr if there is content, to avoid unnecessary empty logs
        if process.stderr:
            logger.warning(f"{description} stderr:\n{process.stderr}")
        return process.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {description}. Command: {' '.join(command)}")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Failed to execute {description}: {e.stderr}")
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}. Please ensure ffmpeg is installed and in your PATH.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during {description}: {e}", exc_info=True)
        raise

def fragment_mp4_to_fmp4(input_mp4_path: str, output_dir: str, segment_duration_ms: int = 500) -> list[dict]:
    """
    Fragments an MP4 file into fMP4 init and media segments for both video and audio streams
    using FFmpeg's DASH capabilities. These segments are suitable for Media Source Extensions (MSE)
    playback in a web browser.

    Args:
        input_mp4_path (str): Path to the input MP4 file (e.g., generated by SadTalker).
        output_dir (str): Temporary directory where fragmented MP4 files will be saved.
                            This directory will be cleaned up automatically after processing.
        segment_duration_ms (int): Desired duration of each media segment in milliseconds.
                                   Smaller values lead to lower latency but more segments.

    Returns:
        list[dict]: A list of dictionaries, each representing an fMP4 segment.
                    Each dictionary contains:
                    - 'data' (bytes): The raw binary content of the segment.
                    - 'segment_type' (str): "init" for initialization segments, "media" for media segments.
                    - 'stream_type' (str): "video" or "audio" to identify the stream type.
                    - 'segment_number' (int, optional): The sequential number for media segments.
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

    segment_duration_seconds = segment_duration_ms / 1000.0 # Convert milliseconds to seconds for FFmpeg

    output_mpd_path = os.path.join(output_dir, "output.mpd") # FFmpeg will create this manifest

    try:
        _run_command([
            "ffmpeg", "-y",
            "-i", input_mp4_path,
            "-v", "error",
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "dash",
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-adaptation_sets", "id=0,streams=v id=1,streams=a",
            "-seg_duration", str(segment_duration_seconds),
            "-init_seg_name", "init_$RepresentationID$.m4s",
            "-media_seg_name", "segment_$RepresentationID$_$Number%05d$.m4s",
            output_mpd_path
        ], description="FFmpeg DASH fragmentation")
    except Exception as e:
        logger.error(f"FFmpeg fragmentation failed for {input_mp4_path}: {e}")
        raise # Re-raise to ensure calling function handles the error

    segments_to_return = []

    # Map FFmpeg numeric stream IDs to human-readable types
    # Assuming stream 0 is video and stream 1 is audio, which is typical for SadTalker output
    stream_id_map = {
        '0': 'video',
        '1': 'audio'
    }

    # 1. Read and collect initialization segments
    init_segments_paths = glob(os.path.join(output_dir, "init_*.m4s"))
    for init_path in init_segments_paths:
        stream_id_match = re.search(r'init_(\d+)\.m4s', os.path.basename(init_path))
        if stream_id_match:
            ffmpeg_stream_id = stream_id_match.group(1)
            stream_type = stream_id_map.get(ffmpeg_stream_id)
            if stream_type:
                with open(init_path, "rb") as f:
                    segments_to_return.append({
                        "data": f.read(),
                        "segment_type": "init",
                        "stream_type": stream_type
                    })
                logger.info(f"Read init segment: {os.path.basename(init_path)} (Stream: {stream_type}, FFmpeg ID: {ffmpeg_stream_id})")
            else:
                logger.warning(f"Unknown numeric stream ID '{ffmpeg_stream_id}' for init segment: {os.path.basename(init_path)}")
        else:
            logger.warning(f"Could not parse init segment name: {os.path.basename(init_path)}")


    # 2. Read and collect media segments
    media_segment_files = {
        "video": [],
        "audio": []
    }

    media_segment_pattern = re.compile(r'segment_(\d+)_(\d+)\.m4s')

    all_media_paths = glob(os.path.join(output_dir, "segment_*.m4s"))
    for media_path in all_media_paths:
        match = media_segment_pattern.match(os.path.basename(media_path))
        if match:
            ffmpeg_stream_id = match.group(1)
            segment_num = int(match.group(2))
            stream_type = stream_id_map.get(ffmpeg_stream_id)

            if stream_type and stream_type in media_segment_files:
                media_segment_files[stream_type].append((segment_num, media_path))
            else:
                logger.warning(f"Unknown numeric stream ID '{ffmpeg_stream_id}' or invalid stream type '{stream_type}' for media segment: {os.path.basename(media_path)}")
        else:
            logger.warning(f"Could not parse media segment name: {os.path.basename(media_path)}")

    # Sort segments by their number to ensure correct playback order
    for stream_type, paths_list in media_segment_files.items():
        paths_list.sort(key=lambda x: x[0]) # Sort by segment number (x[0])
        for segment_num, media_path in paths_list:
            with open(media_path, "rb") as f:
                segments_to_return.append({
                    "data": f.read(),
                    "segment_type": "media",
                    "stream_type": stream_type,
                    "segment_number": segment_num
                })
            logger.info(f"Read media segment: {os.path.basename(media_path)} (Stream: {stream_type}, Num: {segment_num}, FFmpeg ID: {ffmpeg_stream_id})")

    return segments_to_return


def process_lipsync_job_callback(ch, method, properties, body):
    """
    Callback function to process a message from RabbitMQ.
    This function is executed when a new job message is received from the queue.
    It orchestrates fetching image/audio, running SadTalker inference,
    merging video and audio, fragmenting the output video,
    and publishing segments via Redis Pub/Sub.
    """
    session_id = None
    segment_number = None
    temp_audio_path = None
    temp_sadtalker_video_no_audio_path = None # New variable for SadTalker's raw output
    temp_final_merged_video_path = None      # New variable for the video with audio
    temp_fmp4_dir = None

    try:
        job_data = json.loads(body.decode('utf-8'))

        session_id = job_data.get("session_id")
        image_id = job_data.get("image_id")
        audio_segment_base64 = job_data.get("audio_segment_base64")
        segment_number = job_data.get("segment_number")
        is_final_chunk = job_data.get("is_final_chunk", False)
        enhancer = 'gfpgan' if job_data.get("enhancer", False) else None
        use_cpu = job_data.get("use_cpu", False)

        logger.info(f"Processing job for session {session_id}, audio segment {segment_number} "
                            f"(final_chunk: {is_final_chunk}, enhancer: {enhancer}, use_cpu: {use_cpu})")

        # Fetch the image bytes from Redis using the image_id
        image_bytes = redis_client.get(image_id)
        if not image_bytes:
            logger.error(f"Image with ID {image_id} not found in Redis. Cannot proceed with job for session {session_id}.")
            raise ValueError(f"Image {image_id} not found for session {session_id}.")

        # Decode the base64 encoded audio segment into bytes
        if not audio_segment_base64:
            logger.warning(f"Audio segment for session {session_id}, segment {segment_number} is empty. Skipping inference.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        audio_segment_bytes = base64.b64decode(audio_segment_base64)

        # Save PCM bytes as a valid .wav file for SadTalker input
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            pcm_to_wav(audio_segment_bytes, temp_audio_file.name)
            temp_audio_path = temp_audio_file.name
        logger.info(f"Temporary audio saved to: {temp_audio_path}")

        # Run SadTalker inference to generate the lipsync video segment (THIS IS VIDEO ONLY)
        logger.info(f"Running SadTalker inference for session {session_id}, segment {segment_number}...")
        # SadTalker returns only video bytes
        sadtalker_video_no_audio_bytes = run_lipsync_inference_from_bytes(
            image_bytes=image_bytes,
            audio_bytes=open(temp_audio_path, "rb").read(), # Read audio content again
            enhancer=enhancer,
            use_cpu=use_cpu,
            verbose=False
        )
        logger.info(f"SadTalker inference completed for session {session_id}, segment {segment_number}.")

        # Save SadTalker's raw video output to a temporary MP4 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file_no_audio:
            temp_video_file_no_audio.write(sadtalker_video_no_audio_bytes)
            temp_sadtalker_video_no_audio_path = temp_video_file_no_audio.name
        logger.info(f"Temporary SadTalker output (video only) saved to: {temp_sadtalker_video_no_audio_path}")

        # --- NEW: Merge SadTalker's silent video with the original audio ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_merged_video_file:
            temp_final_merged_video_path = temp_merged_video_file.name

        logger.info(f"Merging video from {temp_sadtalker_video_no_audio_path} and audio from {temp_audio_path} into {temp_final_merged_video_path}")
        _run_command([
            "ffmpeg", "-y",
            "-i", temp_sadtalker_video_no_audio_path,  # Input: SadTalker's video (no audio)
            "-i", temp_audio_path,                     # Input: Original audio
            "-c:v", "copy",                            # Copy video stream without re-encoding
            "-c:a", "aac",                             # Encode audio to AAC
            "-b:a", "128k",                            # Audio bitrate
            "-map", "0:v:0",                           # Map first video stream from first input
            "-map", "1:a:0",                           # Map first audio stream from second input
            "-shortest",                               # End when the shortest input stream ends
            temp_final_merged_video_path               # Output merged file
        ], description="FFmpeg video/audio merge")
        logger.info(f"Video and audio merged into: {temp_final_merged_video_path}")
        # --- END NEW MERGE STEP ---

        # NEW: Save the merged video to the designated output directory
        saved_video_filename = f"session_{session_id}_segment_{segment_number}.mp4"
        final_save_path = os.path.join(SAVE_VIDEO_DIR, saved_video_filename)
        shutil.copy(temp_final_merged_video_path, final_save_path)
        logger.info(f"Merged video segment saved to: {final_save_path}")

        # Create a temporary directory for fMP4 segments and fragment the *merged* video
        with tempfile.TemporaryDirectory() as temp_fmp4_dir_ctx:
            temp_fmp4_dir = temp_fmp4_dir_ctx
            # Use the merged video as input for fragmentation
            fmp4_segments = fragment_mp4_to_fmp4(temp_final_merged_video_path, temp_fmp4_dir)
            logger.info(f"Fragmented MP4 into {len(fmp4_segments)} fMP4 segments for session {session_id}, segment {segment_number}.")

            # --- MODIFIED LOGIC HERE TO CONTROL INIT SEGMENT SENDING ---
            # Publish each fMP4 segment to a Redis Pub/Sub channel
            for seg_data in fmp4_segments:
                seg_type = seg_data["segment_type"]
                stream_type = seg_data["stream_type"]

                # Construct Redis hash field for this stream's init status
                init_sent_field = f"{session_id}:{stream_type}_init_sent"

                # Check if this is an init segment and if we've already sent it for this stream/session
                if seg_type == "init":
                    # Check Redis to see if init segment has already been sent for this session and stream type
                    is_init_already_sent = redis_client.hget(REDIS_SESSION_INIT_STATUS_KEY, init_sent_field) == b'true'

                    if is_init_already_sent:
                        logger.debug(f"Skipping duplicate {stream_type} 'init' segment for session {session_id}.")
                        continue # Skip this segment if init already sent
                    else:
                        # Mark as sent in Redis
                        redis_client.hset(REDIS_SESSION_INIT_STATUS_KEY, init_sent_field, 'true')
                        # Set an expiration for the init status flag to prevent old sessions from cluttering Redis
                        # The expiration is set on the entire hash key, not individual fields.
                        # This means the hash will be deleted after 2 hours if not touched.
                        # For short-lived sessions, this is generally fine.
                        redis_client.expire(REDIS_SESSION_INIT_STATUS_KEY, 7200) # 2 hours
                        logger.info(f"Marked {stream_type} 'init' as sent for session {session_id}.")

                # For any segment (init or media), prepare and publish the message
                segment_message = {
                    "session_id": session_id,
                    # Segment number for the audio chunk that triggered this video generation
                    "audio_input_segment_number": segment_number,
                    # Segment number of the fMP4 media segment itself (0 for init)
                    "video_segment_number": seg_data.get("segment_number", 0),
                    "video_segment_base64": base64.b64encode(seg_data["data"]).decode('utf-8'),
                    "segment_type": seg_type, # "init" or "media"
                    "stream_type": stream_type,  # "video" or "audio"
                    # This flag indicates if the *audio* chunk was the final one.
                    # This tells the client that no more video *resulting from audio input* is expected.
                    "is_final_audio_chunk": is_final_chunk
                }
                # Publish the result to a Redis Pub/Sub channel specific to the session_id
                redis_client.publish(f"lipsync_results:{session_id}", json.dumps(segment_message))
                logger.debug(f"Published {seg_type} {stream_type} segment "
                                 f"for session {session_id}, audio seg {segment_number}, video seg {seg_data.get('segment_number', 'init')}.")
            logger.info(f"All fMP4 segments published for session {session_id}, audio segment {segment_number}.")
            # --- END MODIFIED LOGIC ---

        ch.basic_ack(delivery_tag=method.delivery_tag) # Acknowledge successful processing

    except Exception as e:
        logger.error(f"Error processing job for session {session_id}, segment {segment_number}: {e}", exc_info=True)
        error_message = {
            "session_id": session_id,
            "segment_number": segment_number,
            "error": f"Server Error during video generation: {str(e)}",
            "is_error": True
        }
        # Attempt to publish an error message back to the client via Redis Pub/Sub
        if session_id:
            redis_client.publish(f"lipsync_results:{session_id}", json.dumps(error_message))
            logger.info(f"Published error message for session {session_id}.")
        # Negative Acknowledge the message, telling RabbitMQ not to requeue it (as it's likely a processing error)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    finally:
        # Clean up temporary files regardless of success or failure
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info(f"Cleaned up {temp_audio_path}")
        # if temp_video_path and os.path.exists(temp_video_path):
        #   os.remove(temp_video_path)
        #   logger.info(f"Cleaned up {temp_video_path}")
        # temp_fmp4_dir is cleaned up by the TemporaryDirectory context manager.
        if temp_sadtalker_video_no_audio_path and os.path.exists(temp_sadtalker_video_no_audio_path):
            os.remove(temp_sadtalker_video_no_audio_path)
            logger.info(f"Cleaned up {temp_sadtalker_video_no_audio_path}")
        # temp_final_merged_video_path is now copied, so it's safe to remove the temporary version
        if temp_final_merged_video_path and os.path.exists(temp_final_merged_video_path):
            os.remove(temp_final_merged_video_path)
            logger.info(f"Cleaned up temporary merged video: {temp_final_merged_video_path}")


def main():
    global rabbitmq_connection, rabbitmq_channel, _worker_running

    if _worker_running:
        logger.warning("RabbitMQ worker is already running. Skipping initialization.")
        return

    logger.info("Starting RabbitMQ worker...")
    try:
        connection_params = pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
        rabbitmq_connection = pika.BlockingConnection(connection_params)
        rabbitmq_channel = rabbitmq_connection.channel()
        rabbitmq_channel.queue_declare(queue=RABBITMQ_QUEUE_NAME, durable=True)

        rabbitmq_channel.basic_qos(prefetch_count=1) # Process one message at a time

        rabbitmq_channel.basic_consume(
            queue=RABBITMQ_QUEUE_NAME,
            on_message_callback=process_lipsync_job_callback,
            auto_ack=False # We manually acknowledge messages
        )
        _worker_running = True
        logger.info(f"RabbitMQ worker initialized. Waiting for jobs on queue: {RABBITMQ_QUEUE_NAME}...")

        def signal_handler(signum, frame):
            """Graceful shutdown handler for SIGTERM and SIGINT."""
            logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            global _worker_running
            _worker_running = False
            if rabbitmq_connection and not rabbitmq_connection.is_closed:
                try:
                    # Request connection close from a thread-safe callback
                    rabbitmq_connection.add_callback_threadsafe(rabbitmq_connection.close)
                    logger.info("RabbitMQ connection close requested via threadsafe callback.")
                except Exception as e:
                    logger.error(f"Error requesting RabbitMQ connection close: {e}")

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers for SIGTERM and SIGINT registered.")

        logger.info("Calling rabbitmq_channel.start_consuming()...")
        try:
            rabbitmq_channel.start_consuming()
        except pika.exceptions.ConnectionClosedByBroker:
            logger.warning("RabbitMQ connection closed by broker. Worker stopping.")
        except pika.exceptions.AMQPChannelError as e:
            logger.error(f"RabbitMQ Channel Error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during RabbitMQ consumption: {e}", exc_info=True)

        logger.info("RabbitMQ consumer loop finished.")

    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"FATAL ERROR: Could not connect to RabbitMQ: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during worker initialization or run: {e}", exc_info=True)
    finally:
        # Ensure RabbitMQ connection is closed if still open
        if rabbitmq_connection and not rabbitmq_connection.is_closed:
            try:
                rabbitmq_connection.close()
            except Exception as e:
                logger.error(f"Error closing Rabbitmq connection in finally block: {e}")
        _worker_running = False
        logger.info("Worker run loop exited, _worker_running set to False and connections closed.")


if __name__ == "__main__":
    try:
        main()
        logger.info("ai_worker.py exited cleanly after main().")
    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)