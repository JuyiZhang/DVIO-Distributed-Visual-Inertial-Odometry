using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System;
using System.Runtime.InteropServices;

#if ENABLE_WINMD_SUPPORT
using HL2UnityPlugin;
#endif

public class ResearchModeVideoStream : MonoBehaviour
{

    #region Variables

#if ENABLE_WINMD_SUPPORT
    HL2ResearchMode researchMode;
#endif

    enum DepthSensorMode
    {
        ShortThrow,
        LongThrow,
        None
    };

    [SerializeField] bool enablePointCloud = true;
    [SerializeField] GameObject trackedPerson;
    [SerializeField] private Debugger debugger;
    [SerializeField] private TMPro.TextMeshProUGUI pointCloudLength;
    [SerializeField] private TMPro.TextMeshProUGUI currentPose;

    SpatialAnchorController anchorController;

    TCPClient tcpClient;

    public Image longAbImagePreviewPlane = null;
    private Texture2D longAbImageMediaTexture = null;
    private byte[] longAbImageFrameData = null;

    private bool continuousSend = false;

    private Transform frameCameraTransform;

    private bool updatedPointCloudSent = true;
    private float[] pointCloud = new float[] { };

    private Vector3 cameraPosition = new Vector3();
    private Vector3 cameraFrontDirection = new Vector3();

    

#if ENABLE_WINMD_SUPPORT
    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    #endregion

    #region Unity Lifecycle
    private void Awake()
    {
#if UNITY_2020_1_OR_NEWER // note: Unity 2021.2 and later not supported
#if ENABLE_WINMD_SUPPORT
        IntPtr WorldOriginPtr = UnityEngine.XR.WindowsMR.WindowsMREnvironment.OriginSpatialCoordinateSystem;
        unityWorldOrigin = Marshal.GetObjectForIUnknown(WorldOriginPtr) as Windows.Perception.Spatial.SpatialCoordinateSystem;
        unityWorldOrigin = Windows.Perception.Spatial.SpatialLocator.GetDefault().CreateStationaryFrameOfReferenceAtCurrentLocation().CoordinateSystem;
#endif
#else
#if ENABLE_WINMD_SUPPORT
        IntPtr WorldOriginPtr = UnityEngine.XR.WSA.WorldManager.GetNativeISpatialCoordinateSystemPtr();
        unityWorldOrigin = Marshal.GetObjectForIUnknown(WorldOriginPtr) as Windows.Perception.Spatial.SpatialCoordinateSystem;
#endif
#endif
    }

    void Start()
    {
        

        debugger.SetIndicatorState("send", "ip", "Waiting to send point cloud");

        if (longAbImagePreviewPlane != null)
        {
            longAbImageMediaTexture = new Texture2D(320, 288, TextureFormat.Alpha8, false);
            longAbImagePreviewPlane.sprite = Sprite.Create(longAbImageMediaTexture, new Rect(0, 0, 320, 288), new Vector2(160, 144));
        }

        tcpClient = GetComponent<TCPClient>();
        Debug.Log("Research Mode Init");
#if ENABLE_WINMD_SUPPORT
        researchMode = new HL2ResearchMode();
        Debug.Log("Init Long Depth Sensor");
        // Initialize Long Depth Sensor
        researchMode.InitializeLongDepthSensor();
        Debug.Log("Set Reference System");
        researchMode.SetReferenceCoordinateSystem(unityWorldOrigin);
        researchMode.SetPointCloudDepthOffset(0);
        Debug.Log("Start Sensor Loop");
        // Depth sensor should be initialized in only one mode
        researchMode.StartLongDepthSensorLoop(enablePointCloud);

#endif

        Debug.Log("Successfully initiated Hololens Researchmode");

        tcpClient.ConnectToServerEvent();
        anchorController = GetComponent<SpatialAnchorController>();
        anchorController.onAnchorLocationFound += startContinuousSend;
        tcpClient.transformationDataReceived += applyTranformData;
        debugger.toggleContinuousSendSwitch();
    }

    bool startRealtimePreview = true;
    void LateUpdate()
    {
#if ENABLE_WINMD_SUPPORT
        // update long-throw AbImage texture
        if (startRealtimePreview && longAbImagePreviewPlane != null && researchMode.LongAbImageTextureUpdated())
        {
            byte[] frameTexture = researchMode.GetLongAbImageTextureBuffer();
            if (frameTexture.Length > 0)
            {
                if (longAbImageFrameData == null)
                {
                    longAbImageFrameData = frameTexture;
                }
                else
                {
                    System.Buffer.BlockCopy(frameTexture, 0, longAbImageFrameData, 0, longAbImageFrameData.Length);
                }
                longAbImageMediaTexture.LoadRawTextureData(longAbImageFrameData);
                longAbImageMediaTexture.Apply();
                longAbImagePreviewPlane.sprite = Sprite.Create(longAbImageMediaTexture, new Rect(0, 0, 320, 288), new Vector2(160, 144));
            }
        }

        // Update point cloud
        UpdatePointCloud();
#endif
        if (tcpClient.Connected && !updatedPointCloudSent && continuousSend)
        {
            SendLongDepthSensorCombined();
            updatedPointCloudSent = true;
        }
        pointCloudLength.text = "Data Length: " + pointCloud.Length.ToString();
        currentPose.text = cameraPosition.ToString();
    }

    #endregion

#if ENABLE_WINMD_SUPPORT
    private void UpdatePointCloud()
    {
        if (enablePointCloud)
        {
            if (researchMode.LongThrowPointCloudUpdated()){
                pointCloud = researchMode.GetLongThrowPointCloudBuffer();
                
                float[] headPos = researchMode.GetHeadPosition();
                float[] headFwdDir = researchMode.GetHeadForwardVector();
                cameraPosition.Set(headPos[0], headPos[1], headPos[2]);
                cameraFrontDirection.Set(headFwdDir[0], headFwdDir[1], headFwdDir[2]);
                updatedPointCloudSent = false;
            }
        }
    }
#endif



    #region Button Event Functions

    public void SendLongDepthSensorCombined()
    {
        Vector3 anchorPosition = anchorController.getAnchorPosition();
        Vector3 anchorEuler = anchorController.getAnchorRotation();
        Vector3 currentPosition = cameraPosition;
        Vector3 currentRotation = cameraFrontDirection;

#if WINDOWS_UWP
        long timestamp = GetCurrentTimestampUnix();
        var depthMap = researchMode.GetLongDepthMapBuffer();
        if (tcpClient != null)
        {
            tcpClient.SendLongDepthSensorCombined(depthMap, longAbImageFrameData, pointCloud, timestamp, currentRotation, currentPosition, anchorEuler, anchorPosition);
        }
#endif
    }
    public void StopSensorsEvent()
    {
#if ENABLE_WINMD_SUPPORT
        researchMode.StopAllSensorDevice();
#endif
        startRealtimePreview = false;
        Application.Quit();
    }

    public void RestartApplication()
    {
        System.Diagnostics.Process.Start(Application.dataPath.Replace("_Data", "exe"));
    }

    public void startContinuousSend()
    {
        Debug.Log("The continuous send is now " + continuousSend.ToString());
        
        continuousSend = !continuousSend;
        
        if (continuousSend)
        {
            debugger.SetIndicatorState("send", "ok", "Sending Point Cloud");
        } else
        {
            debugger.SetIndicatorState("send", "other", "Continuous Send Paused");
        }
    }

#endregion
    private void OnApplicationFocus(bool focus)
    {
        if (!focus) StopSensorsEvent();
    }

    private float[] PointCloudFromLensToAnchor(float[] pointCloud, Transform anchorTransform)
    {
        var length = pointCloud.Length;
        var transform = anchorTransform.worldToLocalMatrix;
        for(int i=0; i<length/3; i++)
        {
            Vector3 position = new Vector3();
            position.x = pointCloud[i * 3];
            position.y = pointCloud[i * 3 + 1];
            position.z = pointCloud[i * 3 + 2];
            Vector3 localPos = transform.MultiplyPoint(position);
            pointCloud[i * 3] = localPos.x;
            pointCloud[i * 3 + 1] = localPos.y;
            pointCloud[i * 3 + 1] = localPos.z;
        }
        return pointCloud;
    }

    private void applyTranformData()
    {
        Vector3 trackPos = tcpClient.getTrackedPosition();
        Quaternion rotQTracked = new Quaternion();
        Vector3 trackRot = tcpClient.getTrackedRotation();
        rotQTracked.eulerAngles = trackRot;
        //GameObject trackedPerson = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        if (trackedPerson != null)
        {
            trackedPerson.transform.position = trackPos;
            trackedPerson.transform.SetPositionAndRotation(trackPos, rotQTracked);
        }

    }

#if WINDOWS_UWP
    private long GetCurrentTimestampUnix()
    {
        // Get the current time, in order to create a PerceptionTimestamp. 
        Windows.Globalization.Calendar c = new Windows.Globalization.Calendar();
        Windows.Perception.PerceptionTimestamp ts = Windows.Perception.PerceptionTimestampHelper.FromHistoricalTargetTime(c.GetDateTime());
        return ts.TargetTime.ToUnixTimeMilliseconds();
        //return ts.SystemRelativeTargetTime.Ticks;
    }
    private Windows.Perception.PerceptionTimestamp GetCurrentTimestamp()
    {
        // Get the current time, in order to create a PerceptionTimestamp. 
        Windows.Globalization.Calendar c = new Windows.Globalization.Calendar();
        return Windows.Perception.PerceptionTimestampHelper.FromHistoricalTargetTime(c.GetDateTime());
    }
#endif
}