import os
import glob
from pathlib import Path
from tqdm import tqdm
from google.cloud import storage


def download_blob(
    source_blob_name: str, destination_file_name: str, project_id: str, bucket_name: str
) -> None:
    """Downloads a blob from the bucket."""

    storage_client = storage.Client(project=project_id)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def download_gcs_to_local_directory(
    source_path: str, project_id: str, bucket_name: str
) -> None:
    """Downloads a blob from the bucket."""

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_path)
    for blob in tqdm(blobs):
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name)


def list_blobs(source_path: str, project_id: str, bucket_name: str) -> None:
    """Downloads a blob from the bucket."""

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_path)
    for blob in blobs:
        print(blob.name)


def upload_local_directory_to_gcs(
    local_path: str, gcs_path: str, project_id: str, bucket_name: str
) -> None:
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + "/**"):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(
                local_file, gcs_path + "/" + os.path.basename(local_file)
            )
        else:
            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(bucket_name)
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def upload_blob(
    local_path: str, gcs_path: str, project_id: str, bucket_name: str
) -> None:
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)