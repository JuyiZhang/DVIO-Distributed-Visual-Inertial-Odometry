using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;


public class GameScenePreviewManager : MonoBehaviour
{
    [SerializeField]
    private GameObject sceneRoot;

    [SerializeField]
    private GameObject miniMap;

    [SerializeField]
    private Sprite userIcon;

    [SerializeField]
    private GameObject personPrefab;

    [SerializeField]
    private Slider posXSlider;

    [SerializeField]
    private Slider rotYSlider;

    [SerializeField]
    private Slider posZSlider;

    [SerializeField]
    private Debugger debugger;

    private UDPServer udpServer;
    private Dictionary<int, GameObject> personObject2DDict = new Dictionary<int, GameObject>();
    private Dictionary<int, GameObject> personObjectDict = new Dictionary<int, GameObject>();
    private Dictionary<int, int> lastFrameExist = new Dictionary<int, int>();
    private int receivedFrameCount = 0;

    private bool transformInProgress = false;
    private Vector3 scenePos;
    private float sceneRotY;

    private Vector3 scenePosOld;
    private Vector3 sceneRotOld;


    private void Start()
    {
        udpServer = new UDPServer();
        udpServer.coordinateObservationReceived += SpawnPerson;
        sceneRoot.SetActive(false);
    }

    private void Update()
    {
        var user_position = Camera.main.transform.position;
        miniMap.transform.position = new Vector3(-user_position.x * 30, -user_position.z * 30);
    }

    public void SpawnPerson()
    {
        Debug.Log("Spawning person");
        receivedFrameCount++;
        var person_coordinate = udpServer.GetObservedPosition();
        foreach (var person in person_coordinate.Keys)
        {
            if (!personObjectDict.ContainsKey(person))
            {
                Debug.Log("New Person");
                lastFrameExist.Add(person, receivedFrameCount);
                personObjectDict.Add(person, Instantiate(personPrefab));
            }

            Debug.Log("Update Person Status");
            lastFrameExist[person] = receivedFrameCount;
            personObjectDict[person].transform.position = person_coordinate[person].GetPos();
        }
        foreach (var person in lastFrameExist.Keys)
        {
            if (lastFrameExist[person] + 10 < receivedFrameCount) // If 10 consecutive frame is not found for id, then regard as missing
            {
                Debug.Log("Delete Person");
                lastFrameExist.Remove(person);
                personObjectDict.Remove(person);
            }
        }
        
    }

    public void ToggleGamePreviewStatus()
    {
        sceneRoot.SetActive(!sceneRoot.activeSelf);
        Debug.Log("The game preview is set to " + sceneRoot.activeSelf.ToString());
    }

    public void SetSceneTransform()
    {
        Debug.Log("Setting scene transform");
        if (!transformInProgress)
        {
            scenePosOld = sceneRoot.transform.position;
            sceneRotOld = sceneRoot.transform.rotation.eulerAngles;
            transformInProgress = true;
        }
        sceneRotY = sceneRotOld.y + ((rotYSlider.value - 0.5f) * 60f);
        scenePos = scenePosOld + new Vector3(posXSlider.value - 0.5f, 0f, posZSlider.value - 0.5f);
        sceneRoot.transform.SetPositionAndRotation(scenePos, Quaternion.Euler(new Vector3(sceneRotOld.x, sceneRotY, sceneRotOld.z)));
    }

    public void CancelSceneTransform()
    {
        sceneRoot.transform.SetPositionAndRotation(scenePosOld, Quaternion.Euler(sceneRotOld));
        transformInProgress = false;
        debugger.CloseConfigWindow();
    }

}
