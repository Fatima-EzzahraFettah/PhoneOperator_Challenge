"""Fetcher for RAMP data stored in OSF
To adapt it for another challenge, change the CHALLENGE_NAME and upload
public/private data as `tar.gz` archives in dedicated OSF folders named after
the challenge.
"""
import tarfile
import argparse
from zlib import adler32
from pathlib import Path
from osfclient.api import OSF
from osfclient.exceptions import UnauthorizedException
import numpy as np
from PIL import Image
import urllib
import gdown 
import zipfile
import os
import shutil

LOCAL_DATA = Path(__file__).parent / "data"

CHALLENGE_NAME = "phone_operator_churn_challenge"
# you might choosing checking for the correct checksum, if not set
# data_checksum to None
RAMP_FOLDER_CONFIGURATION = {
    "public": dict(
        code="t4uf8",
        archive_name="phone_operator_churn.zip",
        url = "https://drive.google.com/uc?id=1wue-Cyp5OgURqEWfe8Yuu9_YNP2jDwGH",
        # to findout checksum use function
        # defined below: hash_folder(folder_path)
        data_checksum=None,
    ),
    "private": dict(
        code="mqsyhv5d", archive_name="private.tar.gz", data_checksum=None
    ),
}


def get_connection_info(get_private, username=None, password=None):
    "Get connection to OSF and info relative to public/private data."
    if get_private:
        osf, folder_name = OSF(username=username, password=password), "private"
    else:
        assert username is None and password is None, (
            "Username and password should only be provided when fetching "
            "private data."
        )
        osf, folder_name = OSF(), "public"
    data_config = RAMP_FOLDER_CONFIGURATION[folder_name]

    try:
        project = osf.project(data_config["code"])
        store = project.storage("osfstorage")
    except UnauthorizedException:
        raise ValueError("Invalid credentials for RAMP private storage.")
    return store, data_config


def get_one_element(container, name):
    "Get one element from OSF container with a comprehensible failure error."
    elements = [f for f in container if f.name == name]
    container_name = container.name if hasattr(container, "name") else CHALLENGE_NAME
    assert len(elements) == 1, (
        f"There is no element named {name} in {container_name} from the RAMP "
        "OSF account."
    )
    return elements[0]


def hash_folder(folder_path):
    """Return the Adler32 hash of an entire directory."""
    folder = Path(folder_path)

    # Recursively scan the folder and compute a checksum
    checksum = 1
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            checksum = adler32(f.read_bytes(), checksum)
        else:
            checksum = adler32(f.name.encode(), checksum)

    return checksum


def checksum_data(private, raise_error=False):
    folder = "private" if private else "public"
    data_checksum = RAMP_FOLDER_CONFIGURATION[folder]["data_checksum"]
    if data_checksum:
        local_checksum = hash_folder(LOCAL_DATA)
        if raise_error and data_checksum != local_checksum:
            raise ValueError(
                f"The checksum does not match. Expecting {data_checksum} but "
                f"got {local_checksum}. The archive seems corrupted. Try to "
                f"remove {LOCAL_DATA} and re-run this command."
            )

        return data_checksum == local_checksum
    else:
        True


def download_from_osf(private, username=None, password=None):
    "Download and uncompress the data from OSF."

    # check if data directory is empty
    if not LOCAL_DATA.exists() or not any(LOCAL_DATA.iterdir()):
        LOCAL_DATA.mkdir(exist_ok=True)

        print("Checking the data URL...", end="", flush=True)
        # Get the connection to OSF
        store, data_config = get_connection_info(
            private, username=username, password=password
        )
        archive_name = data_config["archive_name"]

        
        '''

        # Find the folder in the OSF project
        challenge_folder = get_one_element(store.folders, CHALLENGE_NAME)

        osf_file = get_one_element(challenge_folder.files, archive_name)

        # Find the file to download from the OSF project
        print("Ok.")
        '''
        ##############
        
        url = data_config["url"]
       
        ARCHIVE_PATH = LOCAL_DATA / archive_name
        gdown.download(url, './data/' + archive_name, quiet=False)
        
        # Uncompress the data in the data folder
        print("Extracting now...", end="", flush=True)
        with zipfile.ZipFile(ARCHIVE_PATH) as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
            
                # copy file (taken from zipfile's extract)
                source = zip_file.open(member)
                target = open(os.path.join(LOCAL_DATA, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

       
      

        # Clean the directory by removing the archive
        print("Removing the archive...", end="", flush=True)
        ARCHIVE_PATH.unlink()
      
        print("Checking the data...", end="", flush=True)
        checksum_data(private, raise_error=True)
    
    else:
        print(
            f"{LOCAL_DATA} directory is not empty. Please empty it or select"
            " another destination for LOCAL_DATA if you wish to proceed"
        )


def jpg_to_npy(image):
    image_name = image.split(".")[0]
    im = Image.open(image)
    im = np.array(im)
    np.save(f"{image_name}.npy", im)

def get_stringency(url_):
    url = url_
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Data loader for the {CHALLENGE_NAME} challenge on RAMP."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If this flag is used, download the private data "
        "from OSF. This requires the username and password "
        "options to be provided.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Username for downloading private OSF data.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password for downloading private OSF data.",
    )
    args = parser.parse_args()
    download_from_osf(args.private, args.username, args.password)
