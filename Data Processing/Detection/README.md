# DVIO - Detection
## Function
The function of the component is to

1. Receive the camera data obtained from Unity App in Data Collection section
2. Save the collected data
3. Detect the 2D coordinate of the keypoint of person in the camera space
4. Convert the camera space coordinates of each keypoint to 3D coordinates
5. Obtain the person's location and pose based on camera space coordinates
6. Perform step 3-5 in offline mode
7. Send the data of step 5 back to the data collection section or to data visualization section for visualizing the tracking

## Environment
It is recommended to use Linux if pose detection is used as mediapipe does not support GPU inferrence in Windows
The app assumes that Python is installed. The developer uses Python 3.11.6 and 3.8.0 without issue

## Usage
1. Install all requirements

> For M-series SoC and CPU Inferrence
> `python -m pip install -r requirements.txt`

> For NVIDIA GPU Inferrence
> `python -m pip install -r requirements-cu118.txt`

2. Run the backend GUI

>`python main.py`

3. To collect data from data collection app, click "Start TCP Server" button
4. To save data, click File->Save Raw Data
5. The data is stored in `*/data/Session_{Timestamp of session}`
6. To view previous collected data, open `Process` Tab
7. The data is automatically listed in the dropdown list, you can also browse other location 
