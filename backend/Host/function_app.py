import azure.functions as func
import logging
import os
import json
import time
from azure.storage.blob import BlobClient
from azure.cosmos import CosmosClient
import azure.cognitiveservices.speech as speechsdk
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

app = func.FunctionApp()

# Retrieve secrets from Key Vault using Managed Identity
def get_secret_from_key_vault(secret_name):
    try:
        # Use DefaultAzureCredential which chains ManagedIdentityCredential
        credential = DefaultAzureCredential()
        # Replace with your Key Vault URI
        key_vault_uri = os.environ["KEY_VAULT_URI"]
        secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
        secret = secret_client.get_secret(secret_name)
        return secret.value
    except Exception as e:
        logging.error(f"Error retrieving secret {secret_name} from Key Vault: {e}")
        # Fallback to environment variables for local development or if Managed Identity fails
        return os.environ.get(secret_name.replace('-', '_').upper()) # e.g., SPEECH_SERVICE_KEY

# Global clients (initialized once)
SPEECH_KEY = get_secret_from_key_vault("SPEECH-SERVICE-KEY")
SPEECH_REGION = os.environ.get("SPEECH_SERVICE_REGION") # This can be an environment variable
COSMOSDB_URI = get_secret_from_key_vault("COSMOSDB-URI")
COSMOSDB_PRIMARY_KEY = get_secret_from_key_vault("COSMOSDB-PRIMARY-KEY")
STORAGE_CONNECTION_STRING = get_secret_from_key_vault("STORAGE-CONNECTION-STRING")

COSMOS_CLIENT = None
COSMOS_DB_NAME = "transcriptions"
COSMOS_CONTAINER_NAME = "jobs"

def get_cosmos_client():
    global COSMOS_CLIENT
    if COSMOS_CLIENT is None:
        COSMOS_CLIENT = CosmosClient(COSMOSDB_URI, credential={"masterKey": COSMOSDB_PRIMARY_KEY})
    return COSMOS_CLIENT

@app.blob_trigger(arg_name="myblob", path="audio-uploads/{name}",
                  connection="AzureWebJobsStorage") # Use the default storage account for the trigger
def transcribe_audio_trigger(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob Name: {myblob.name}, Size: {myblob.length} Bytes")
    blob_name = myblob.name.split('/')[-1]
    job_id = str(time.time()).replace('.', '') + "_" + blob_name # Simple unique ID for now

    try:
        # Access the source blob directly using BlobClient (Managed Identity or connection string)
        source_blob_client = BlobClient.from_connection_string(
            conn_str=STORAGE_CONNECTION_STRING,
            container_name="audio-uploads",
            blob_name=blob_name
        )
        source_blob_url = source_blob_client.url

        # Speech configuration
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.set_service_property(
            name="diarizationEnabled",
            value="true",
            channel=speechsdk.ServicePropertyChannel.UriQueryParameter
        )
        speech_config.set_service_property(
            name="conversationTranscription",
            value="true",
            channel=speechsdk.ServicePropertyChannel.UriQueryParameter
        )
        audio_config = speechsdk.AudioConfig(url=source_blob_url)
        conversation_transcriber = speechsdk.ConversationTranscriber(
            speech_config=speech_config, audio_config=audio_config
        )

        # Store job details in Cosmos DB
        cosmos_client = get_cosmos_client()
        database = cosmos_client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(COSMOS_CONTAINER_NAME)
        container.upsert_item({
            "id": job_id,
            "filename": blob_name,
            "status": "Processing",
            "timestamp": time.time(),
            "transcription": [],
            "source_blob_url": source_blob_url # Store for deletion later
        })

        transcription_result = []

        def cb_recognized(evt: speechsdk.SpeechRecognitionEventArgs):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                json_result = json.loads(evt.result.json)
                # The 'NBest' array contains detailed information for diarization
                for item in json_result.get("NBest", []):
                    if "SpeakerId" in item and "Display" in item:
                        transcription_result.append({
                            "speaker": f"Speaker {item['SpeakerId']}",
                            "text": item['Display'],
                            "offset": item.get("Offset", 0), # in ticks (100 nanoseconds)
                            "duration": item.get("Duration", 0) # in ticks
                        })
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                logging.info(f"No speech could be recognized: {evt.result.no_match_details}")

        def cb_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
            logging.warning(f"Transcription canceled: {evt.reason}")
            if evt.reason == speechsdk.ResultReason.CanceledReason.Error:
                logging.error(f"Canceled due to error: {evt.error_details}")
            # Update status in Cosmos DB to Canceled/Error
            container.patch_item(
                item=job_id,
                partition_key=job_id,
                operations=[
                    {"op": "replace", "path": "/status", "value": "Error"},
                    {"op": "replace", "path": "/error_details", "value": evt.error_details}
                ]
            )

        def cb_session_stopped(evt: speechsdk.SessionEventArgs):
            logging.info(f"Session stopped event for job {job_id}.")
            # Update status and transcription in Cosmos DB
            container.patch_item(
                item=job_id,
                partition_key=job_id,
                operations=[
                    {"op": "replace", "path": "/status", "value": "Completed"},
                    {"op": "replace", "path": "/transcription", "value": transcription_result}
                ]
            )
            # Delete the audio file from blob storage after successful transcription
            try:
                source_blob_client.delete_blob()
                logging.info(f"Deleted source audio file: {blob_name}")
            except Exception as e:
                logging.error(f"Error deleting blob {blob_name}: {e}")

        conversation_transcriber.recognized.connect(cb_recognized)
        conversation_transcriber.canceled.connect(cb_canceled)
        conversation_transcriber.session_stopped.connect(cb_session_stopped)

        logging.info(f"Starting batch conversation transcription for {blob_name} with job ID {job_id}...")
        conversation_transcriber.start_transcribing_async().get()

        # The transcription process is asynchronous, results are handled by callbacks
        # We don't need a loop here as callbacks will update Cosmos DB
        # The function will complete, but the transcription will continue in the Speech Service
    except Exception as e:
        logging.error(f"Error processing blob {blob_name}: {e}")
        # Ensure error status is captured in Cosmos DB
        cosmos_client = get_cosmos_client()
        database = cosmos_client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(COSMOS_CONTAINER_NAME)
        container.upsert_item({
            "id": job_id,
            "filename": blob_name,
            "status": "Error",
            "timestamp": time.time(),
            "error_details": str(e)
        })

@app.route(route="getTranscription/{jobId}", auth_level=func.AuthLevel.FUNCTION)
def get_transcription(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get transcription.')
    job_id = req.route_params.get('jobId')

    if not job_id:
        return func.HttpResponse(
            "Please pass a jobId in the route.",
            status_code=400
        )

    try:
        cosmos_client = get_cosmos_client()
        database = cosmos_client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(COSMOS_CONTAINER_NAME)
        item = container.read_item(item=job_id, partition_key=job_id)

        return func.HttpResponse(
            json.dumps(item),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error retrieving transcription for job ID {job_id}: {e}")
        return func.HttpResponse(
            f"Error retrieving transcription: {e}",
            status_code=500
        )