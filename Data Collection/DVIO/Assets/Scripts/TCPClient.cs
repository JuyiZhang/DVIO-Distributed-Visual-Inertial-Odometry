using System;
using System.Threading;
using UnityEngine;
//using System.Runtime.Serialization.Formatters.Binary;
#if WINDOWS_UWP
using Windows.Networking.Sockets;
using Windows.Storage.Streams;
#endif

public class TCPClient : MonoBehaviour
{
    [SerializeField]
    private string port;

    [SerializeField]
    private Debugger debugger;

    [SerializeField]
    private TMPro.TextMeshProUGUI buttonText; 

    private string hostIPAddress;

    private bool connected = false;
    public bool Connected
    {
        get { return connected; }
    }
    private Thread dataRcvThread;
    private NetworkUtils networkUtils;
    public event Notify transformationDataReceived;
    private Vector3 trackedPosition = new Vector3();
    private Vector3 trackedRotation = new Vector3();

    #region Unity Functions


    private void Awake()
    {
        debugger.SetIndicatorState("tcp", "ip", "Pending Connection");
    }

    private void Start()
    {
        networkUtils = GetComponent<NetworkUtils>();
        networkUtils.onHostIPRetrieved += OnHostIPFound;
    }

    private void OnApplicationFocus(bool focus)
    {
        if (!focus)
        {
            StopConnection();
        }
    }
    #endregion // Unity Functions

    

#if WINDOWS_UWP
    StreamSocket socket = null;
    public DataWriter dw;
    public DataReader dr;
#endif

    private async void StartConnection()
    {
        Debug.Log("Starting Connection...");

#if WINDOWS_UWP
        if (socket != null) socket.Dispose();

        try
        {
            socket = new StreamSocket();
            var hostName = new Windows.Networking.HostName(hostIPAddress);
            await socket.ConnectAsync(hostName, port);
            dw = new DataWriter(socket.OutputStream);
            dr = new DataReader(socket.InputStream);
            dr.InputStreamOptions = InputStreamOptions.Partial;
            connected = true;
            buttonText.text = "Disconnect";
            debugger.SetIndicatorState("tcp", "ok", "Connected to " + hostIPAddress);

            dataRcvThread = new Thread(new ThreadStart(dataRcv));
            dataRcvThread.IsBackground = true;
            dataRcvThread.Start();
        }
        catch (Exception ex)
        {
            SocketErrorStatus webErrorStatus = SocketError.GetStatus(ex.GetBaseException().HResult);
            Debug.Log("Stream Error Detected");
            Debug.Log(webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
            debugger.SetIndicatorState("tcp", "error", webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
        }
#endif
    }

    private async void dataRcv()
    {
        Byte[] bytes = new Byte[sizeof(float) * 6];
        while (true)
        {
#if WINDOWS_UWP
            await dr.LoadAsync(sizeof(float) * 6);
            dr.ReadBytes(bytes);
#endif
            float[] receivedPose = BytesToFloat(bytes);
            Debug.Log("Data Received: " + receivedPose.Length.ToString());
            if(receivedPose.Length == 6) {
                trackedPosition.Set(receivedPose[0], receivedPose[1], receivedPose[2]);
                trackedRotation.Set(receivedPose[3], receivedPose[4], receivedPose[5]);
                Debug.Log("Position: " + trackedPosition.ToString() + ", Rotation: " + trackedRotation.ToString());
                transformationDataReceived?.Invoke();
            }
            
        }
    }

    private void StopConnection()
    {
        buttonText.text = "Connect";
#if WINDOWS_UWP
        dw?.DetachStream();
        dw?.Dispose();
        dw = null;

        dr?.DetachStream();
        dr?.Dispose();
        dr = null;

        socket?.Dispose();
        connected = false;
#endif
    }

#if WINDOWS_UWP
    #region Send Data
    
    bool lastMessageSent = true;

    public async void SendLongDepthSensorCombined(ushort[] data1, byte[] data2, float[] pointCloud, long timestamp, Vector3 euler, Vector3 position, Vector3 anchorEuler, Vector3 anchorPosition){
    
        if (!lastMessageSent) return;
        lastMessageSent = false;
        try
        {
            float[] transform = {position.x, position.y, position.z, euler.x, euler.y, euler.z, anchorPosition.x, anchorPosition.y, anchorPosition.z, anchorEuler.x, anchorEuler.y, anchorEuler.z};
            // Write header
            dw.WriteString("c"); // header "c" 

            // Write timestamp
            dw.WriteInt64(timestamp);
            dw.WriteInt32(data1.Length * 2);
            dw.WriteInt32(data2.Length);
            dw.WriteInt32(pointCloud.Length * 4 + 48);
            dw.WriteBytes(UINT16ToBytes(data1));
            dw.WriteBytes(data2);
            dw.WriteBytes(FloatToBytes(pointCloud));
            dw.WriteBytes(FloatToBytes(transform));
            

            // Send out
            await dw.StoreAsync();
            await dw.FlushAsync();
        }
        catch (Exception ex)
        {
            SocketErrorStatus webErrorStatus = SocketError.GetStatus(ex.GetBaseException().HResult);
            Debug.Log(webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
            debugger.SetIndicatorState("tcp", "error", webErrorStatus.ToString() != "Unknown" ? webErrorStatus.ToString() : ex.Message);
            StopConnection();
        }
        lastMessageSent = true;
    
    }
    #endregion
#endif

    #region Public Function
    public Vector3 getTrackedPosition()
    {
        return trackedPosition;
    }
    public Vector3 getTrackedRotation()
    {
        return trackedRotation;
    }
    #endregion

    #region Conversion Function
    byte[] UINT16ToBytes(ushort[] data)
    {
        byte[] ushortInBytes = new byte[data.Length * sizeof(ushort)];
        System.Buffer.BlockCopy(data, 0, ushortInBytes, 0, ushortInBytes.Length);
        return ushortInBytes;
    }

    byte[] FloatToBytes(float[] data)
    {
        byte[] floatInBytes = new byte[data.Length * sizeof(float)];
        System.Buffer.BlockCopy(data, 0, floatInBytes, 0, floatInBytes.Length);
        return floatInBytes;
    }

    float[] BytesToFloat(byte[] data)
    {
        float[] bytesInFloat = new float[data.Length / sizeof(float)];
        System.Buffer.BlockCopy(data, 0, bytesInFloat, 0, data.Length);
        return bytesInFloat;
    }
#endregion

#region Button Callback
    public void ConnectToServerEvent()
    {
        var networkUtil = GetComponent<NetworkUtils>();
        if (!connected) Debug.Log("Begin Connection...");
        networkUtil.syncHostIP();
    }
#endregion

#region Delegate Callback
    public void OnHostIPFound()
    {
        var newAddress = networkUtils.getHostIP();
        if (hostIPAddress != newAddress)
        {
            Debug.Log("Host IP updated to: " + newAddress);
            hostIPAddress = newAddress;
        }
        if (!connected) StartConnection();
        else StopConnection();
    }
#endregion
}
