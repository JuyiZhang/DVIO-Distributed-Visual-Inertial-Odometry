using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;
using UnityEngine;

public delegate void Notify();

public class NetworkUtils : MonoBehaviour
{
    public event Notify onAnchorUpdate;
    public event Notify onAnchorNotFound;
    public event Notify onHostIPRetrieved;
    public event Notify onHostIPNotRetrieved;

    [SerializeField]
    private string website = "https://mutaw.azurewebsites.net/";

    

    private AnchorData anchorData = new AnchorData();
    private string hostip = "";

    #region Unity Lifecycle
    // Start is called before the first frame update
    void Start()
    {
        anchorData.creator = SystemInfo.deviceUniqueIdentifier;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    #endregion

    #region Public Functions
    public void setAnchorID(string id)
    {
        anchorData.id = id;
        StartCoroutine(pushAnchorID());
    }

    public void getAnchorID()
    {
        StartCoroutine(pullAnchorID());
    }

    public void syncHostIP()
    {
        StartCoroutine(pullHostIP());
    }

    public AnchorData getAnchorData()
    {
        return anchorData;
    }

    public string getHostIP()
    {
        return hostip;
    }
    #endregion

    #region HTTP Request
    IEnumerator pushAnchorID()
    {
        string dataToSend = JsonUtility.ToJson(anchorData);
        WWWForm formToSend = new WWWForm();
        formToSend.AddField("anchorID",anchorData.id);
        formToSend.AddField("creator", anchorData.creator);
        UnityWebRequest www = UnityWebRequest.Post(website + "add_anchor", formToSend);
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.Log("Request Insuccessful");
        } else
        {
            Debug.Log("Request Successful " + www.downloadHandler.text);
        }
    }

    IEnumerator pullAnchorID()
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(website+"query_anchor"))
        {
            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Request Successful");
                string resultJsonData = webRequest.downloadHandler.text;
                if (resultJsonData.Contains("failed"))
                {
                    // Failure Handler
                    Debug.Log("Anchor Does Not Exist, Proceed with Anchor Creation");
                    onAnchorNotFound?.Invoke();
                } else
                {
                    AnchorResult serverAnchorData = JsonUtility.FromJson<AnchorResult>(resultJsonData);
                    anchorData.id = serverAnchorData.id;
                    anchorData.creator = serverAnchorData.creator;
                    Debug.Log("Request Successful with Anchor ID: " + anchorData.id);
                    onAnchorUpdate?.Invoke();
                }
                
            }
        }
    }

    IEnumerator pullHostIP()
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(website + "query_host"))
        {
            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Request Successful");
                string result = webRequest.downloadHandler.text;
                if (result == "Offline")
                {
                    // Failure Handler
                    Debug.Log("Host Offline");
                    onHostIPNotRetrieved?.Invoke();
                }
                else
                {
                    hostip = result;
                    Debug.Log("Current Host IP is: " + hostip);
                    onHostIPRetrieved?.Invoke();
                }

            }
        }
    }
    #endregion

}

public class AnchorData
{
    public string id;
    public string creator;
}

public class AnchorResult
{
    public string result;
    public string id;
    public string creator;
    public string ctime;
}