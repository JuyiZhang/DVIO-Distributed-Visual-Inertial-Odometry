using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Threading;
using System.Net.Sockets;
using System;

public class UDPServer : MonoBehaviour
{

    public event Notify devicePositionReceived;
    public event Notify coordinateObservationReceived;
    public event Notify poseObservationReceived;

    private Thread UDPServerThread;

    private Dictionary<int, DevicePosition> devicePositions;
    private Dictionary<int, DevicePosition> observedHumanPosition;

    [SerializeField]
    private int port = 8848;

    private Vector3[] offset = { new Vector3(97.285f, -0.061f, 0.349f), new Vector3(0, -90f, 0) };

    private System.DateTime epochStart;
    // Start is called before the first frame update
    void Start()
    {
        epochStart = new System.DateTime(1970, 1, 1, 0, 0, 0, System.DateTimeKind.Utc);

        UDPServerThread = new Thread(new ThreadStart(UDPListenForData));
        UDPServerThread.IsBackground = true;
        UDPServerThread.Start();
        devicePositions = new Dictionary<int, DevicePosition>();
        observedHumanPosition = new Dictionary<int, DevicePosition>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void UDPListenForData()
    {
        IPAddress ipAddress = IPManager.GetIP();
        IPEndPoint incomingConnection = new IPEndPoint(ipAddress, port);
        Socket udpSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        udpSocket.Bind(incomingConnection);
        Debug.Log("Start listening at " + ipAddress.ToString());
        IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);
        EndPoint Remote = (EndPoint)(sender);
        byte[] data = new byte[1024];
        while (true)
        {
            udpSocket.ReceiveFrom(data, ref Remote);
            int dataType = BitConverter.ToInt32(data, 0);
            int dataLength = BitConverter.ToInt32(data, 4);
            Debug.Log("Received data length: " + dataLength.ToString());
            Debug.Log("Received data type: " + dataType.ToString());
            for (int i = 0; i < dataLength; i++)
            {
                if (dataType == 0)
                {
                    int deviceIP = BitConverter.ToInt32(data, i * 52 + 8);
                    Vector3 devicePos = BytesToVector3(data, i * 52 + 12);
                    Vector3 deviceFwdVector = BytesToVector3(data, i * 52 + 24);
                    if (devicePositions.ContainsKey(deviceIP))
                        devicePositions[deviceIP] = new DevicePosition(devicePos, deviceFwdVector);
                    else
                        devicePositions.Add(deviceIP, new DevicePosition(devicePos, deviceFwdVector));
                    Debug.Log("Add device: " + devicePos.ToString());
                }
                else if (dataType == 2)
                {
                    int coordinateID = BitConverter.ToInt32(data, i * 192 + 8);
                    Vector3 coordinatePos = BytesToVector3(data, i * 192 + 12);
                    Quaternion[] coordinateRot = BytesToArmatureQuaternionArray(data, i * 192 + 24);
                    if (observedHumanPosition.ContainsKey(coordinateID))
                        observedHumanPosition[coordinateID] = new DevicePosition(coordinatePos, coordinateRot);
                    else
                        observedHumanPosition.Add(coordinateID, new DevicePosition(coordinatePos, coordinateRot));
                    Debug.Log("Add coordinate: " + coordinatePos.ToString());
                }
                else if (dataType == 1)
                {
                    int coordinateID = BitConverter.ToInt32(data, i * 16 + 8);
                    Vector3 coordinatePos = BytesToVector3(data, i * 16 + 12);
                    if (observedHumanPosition.ContainsKey(coordinateID))
                        observedHumanPosition[coordinateID] = new DevicePosition(coordinatePos, 0.0f);
                    else
                        observedHumanPosition.Add(coordinateID, new DevicePosition(coordinatePos, 0.0f));
                    Debug.Log("Received person ID of " + coordinateID.ToString());
                    Debug.Log("The position is: " + coordinatePos.ToString());
                }
            }
            float timestamp = 0;
            if (dataType == 0)
            {
                timestamp = BitConverter.ToSingle(data, dataLength * 52 + 8);
                float timenow = (float)(DateTime.UtcNow - epochStart).TotalSeconds;
                Debug.Log("Received Device Coordinate with time" + ((timenow - timestamp)*10000).ToString());
                devicePositionReceived?.Invoke();
            }
            else if (dataType == 2)
            {
                timestamp = BitConverter.ToSingle(data, dataLength * 192 + 8);
                float timenow = (float)(DateTime.Now - epochStart).TotalMilliseconds;
                Debug.Log("Received observed coordinate with time" + (timenow - timestamp).ToString());
                poseObservationReceived?.Invoke();
            }
            else if (dataType == 1)
            {
                timestamp = BitConverter.ToSingle(data, dataLength * 16 + 8);
                float timenow = (float)(DateTime.Now - epochStart).TotalMilliseconds;
                Debug.Log("Received observed coordinate with time" + (timenow - timestamp).ToString());
                coordinateObservationReceived?.Invoke();
            }

        }
    }

    Vector3 BytesToVector3(byte[] data, int startIndex)
    {
        float[] array = BytesToFloatArray(data, startIndex, 3);
        return new Vector3(array[0], array[1], array[2]);
    }

    Quaternion[] BytesToArmatureQuaternionArray(byte[] data, int startIndex)
    {
        float[] array = BytesToFloatArray(data, startIndex, 44);
        Quaternion[] armatureRotation = new Quaternion[11];
        for (int i = 0; i < 10; i++)
        {
            armatureRotation[i] = new Quaternion(array[i * 4], array[i * 4 + 1], array[i * 4 + 2], array[i * 4 + 3]);
        }
        return armatureRotation;
    }

    float[] BytesToFloatArray(byte[] data, int startIndex, int arrayLength)
    {
        float[] array = new float[arrayLength];
        for (int i = 0; i < arrayLength; i++)
        {
            array[i] = BitConverter.ToSingle(data, startIndex + i * 4);
        }
        return array;
    }

    public Dictionary<int, DevicePosition> GetDevicePosition()
    {
        return devicePositions;
    }

    public Dictionary<int, DevicePosition> GetObservedPosition()
    {
        return observedHumanPosition;
    }

}

public class DevicePosition
{
    private Vector3 devicePos;
    private Vector3 deviceFwdVector;
    private Quaternion[] deviceArmature;
    private float rotationY;

    public DevicePosition(Vector3 pos, Vector3 fwdVector)
    {
        devicePos = pos;
        deviceFwdVector = fwdVector;
    }

    public DevicePosition(Vector3 pos, Quaternion[] armature)
    {
        devicePos = pos;
        deviceArmature = armature;
    }

    public DevicePosition(Vector3 pos, float rotate)
    {
        devicePos = pos;
        rotationY = rotate;
    }

    public override string ToString()
    {
        if (deviceFwdVector == null)
            return "Position: " + devicePos.ToString() + "Rotation: " + deviceArmature.ToString();
        else
            return "Position: " + devicePos.ToString() + "Forward Vector: " + deviceFwdVector.ToString();
    }

    public Vector3 GetPos()
    {
        return devicePos;
    }
    public Vector3 GetFwdVector()
    {
        return deviceFwdVector;
    }
    public Quaternion[] GetRot()
    {
        return deviceArmature;
    }
    public float GetRotY()
    {
        return rotationY;
    }
}
