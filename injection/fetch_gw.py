import requests
import numpy as np
from urllib.request import urlretrieve
from pycbc.frame import read_frame

####Define functions
def fetch_run_gps_times(run):
    "Fetch gwosc archive and return the (start, end) GPS time tuple of the run."
    response = requests.get("https://gwosc.org/archive/all/json/").json()
    
    runs = response["runs"]
    run_info = runs.get(run)
    if run_info is None:
        raise ValueError(f"Could not find run {run}. Available runs: {runs.keys()}")
    return run_info["GPSstart"], run_info["GPSend"]

def fetch_strain_list(run, detector, gps_start=None, gps_end=None):
    "Return the list of strain file info for `run` and `detector`."
    if gps_start is None or gps_end is None:
        start, end = fetch_run_gps_times(run)
        gps_start = gps_start or start
        gps_end = gps_end or end

    # Get the strain list
    fetch_url = (
        f"https://gwosc.org/archive/links/"
        f"{run}/{detector}/{gps_start}/{gps_end}/json/"
    )
    response = requests.get(fetch_url)
    response.raise_for_status()
    return response.json()["strain"]

def get_gps_start(strain_files):
    start_time_list = []
    for i in range(len(strain_files)):
        start_time_list.append(strain_files[i]['GPSstart'])
    return start_time_list

def fetch_strain(gps_time, detector, dataset, file_frmt="gwf"):
    # Get the download url
    fetch_url = (
        f"https://gwosc.org/archive/links/"
        f"{dataset}/{detector}/{gps_time}/{gps_time}/json/"
    )
    response = requests.get(fetch_url)
    response.raise_for_status()
    json_response = response.json()
    for strain_file in json_response["strain"]:
        if strain_file["detector"] == detector and strain_file["format"] == file_frmt:
            download_url = strain_file["url"]
            filename = download_url[download_url.rfind("/") + 1 :]
            break
    else:
        raise ValueError(f"Strain url not found for detector {detector}.")

    #print(f"Downloading data file")
    #print(download_url, filename)
    urlretrieve(download_url,filename)
    channel_name = str(detector)+":GWOSC-4KHZ_R1_STRAIN"
    data = read_frame(filename, channel_name)
    return data
