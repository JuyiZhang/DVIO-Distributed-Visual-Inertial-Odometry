using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class GameObjectManager : MonoBehaviour
{
    [SerializeField]
    private GameObject devicePrefab;

    [SerializeField]
    private GameObject coordinatePrefab;

    private UDPServer udpServer;
    private Dictionary<int, GameObject> deviceGameObjects;
    private Dictionary<int, GameObject> observedGameObjects;
    private bool updateDevice = false;
    private bool updateObserved = false;
    // Start is called before the first frame update
    void Start()
    {
        udpServer = GetComponent<UDPServer>();
        udpServer.devicePositionReceived += UpdateDevicePosition;
        udpServer.coordinateObservationReceived += UpdateObservedCoordinate;
        deviceGameObjects = new Dictionary<int, GameObject>();
        observedGameObjects = new Dictionary<int, GameObject>();
    }

    // Update is called once per frame
    void Update()
    {
        if (updateDevice)
        {
            var devicePositions = udpServer.GetDevicePosition();
            for (int i = 0; i < devicePositions.Keys.Count; i++)
            {
                int[] deviceKey = devicePositions.Keys.ToArray<int>();
                Debug.Log(deviceKey.Length);
                if (!deviceGameObjects.ContainsKey(deviceKey[i]))
                {
                    deviceGameObjects.Add(deviceKey[i], Instantiate(devicePrefab));
                }
                deviceGameObjects[deviceKey[i]].transform.position = devicePositions[deviceKey[i]].GetPos();
                deviceGameObjects[deviceKey[i]].transform.forward = devicePositions[deviceKey[i]].GetFwdVector();

            }
            updateDevice = false;
        }
        if (updateObserved)
        {
            var observedCoordinatePosition = udpServer.GetObservedPosition();
            for (int i = 0; i < observedCoordinatePosition.Keys.Count; i++)
            {
                int[] deviceKey = observedCoordinatePosition.Keys.ToArray<int>();
                //Debug.Log(deviceKey.Length);
                if (!observedGameObjects.ContainsKey(0))//deviceKey[i]))
                {
                    observedGameObjects.Add(0, Instantiate(coordinatePrefab));
                }
                observedGameObjects[0].transform.position = observedCoordinatePosition[deviceKey[i]].GetPos();
                observedGameObjects[0].transform.eulerAngles = new Vector3(0,observedCoordinatePosition[deviceKey[i]].GetRotY(),0);
                //var observedAvatar = observedGameObjects[0].GetComponent<AvatarController>();
                //observedAvatar.setAllBone(observedCoordinatePosition[deviceKey[i]].GetRot());
            }
        }
    }

    private void UpdateDevicePosition()
    {
        updateDevice = true;
    }

    private void UpdateObservedCoordinate()
    {
        updateObserved = true;
    } 
}
