using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpatialAnchorController : MonoBehaviour
{

    [SerializeField]
    private GameObject anchorObject;

    [SerializeField]
    private Debugger debugger;

    private AnchorModuleScript anchorModule;
    private NetworkUtils networkUtils;

    private Pose anchorTransform;

    public event Notify onAnchorLocationFound;
    public event Notify onAnchorLocationNotFound;

    // Start is called before the first frame update
    void Start()
    {
        debugger.SetIndicatorState("anchor", "ip", "Pending Anchor Init");
        networkUtils = GetComponent<NetworkUtils>();
        networkUtils.onAnchorNotFound += onAnchorIDNotFound;
        networkUtils.onAnchorUpdate += onAnchorIDFound;
        anchorModule = anchorObject.GetComponent<AnchorModuleScript>();
        anchorModule.OnStartASASession += startAzureAfterInit;
    }
    
    // Update is called once per frame
    void Update()
    {
    
    }

    #region Public Functions

    public Vector3 getAnchorRotation()
    {
        return anchorTransform.rotation.eulerAngles;
    }

    public Vector3 getAnchorPosition()
    {
        return anchorTransform.position;
    }

    public Transform getAnchorTransform()
    {
        return anchorObject.transform;
    }
    #endregion

    #region Event Handler

    private void startAzureAfterInit()
    {
        Debug.Log("Init Finished");
        networkUtils.getAnchorID();
    }

    private void onAnchorIDNotFound()
    {
        Debug.Log("Anchor ID not found, proceeed creating anchor...");
        anchorModule.OnCreateAnchorSucceeded += onCreateAnchorSucceeded;
        anchorModule.CreateAzureAnchor(anchorObject);
    }

    private void onAnchorIDFound()
    {
        string anchorID = networkUtils.getAnchorData().id;
        if (anchorID == "")
        {
            Debug.Log("Anchor ID empty, treated as not found");
            onAnchorIDNotFound();
            return;
        }
        Debug.Log("Anchor ID Found to be: " + anchorID);
        anchorModule.OnFoundASAAnchor += onAnchorFound;
        anchorModule.FindAzureAnchor(anchorID);
    }

    private void onAnchorFound()
    {
        Debug.Log(anchorModule.currentAzureAnchorID + " Found");
        anchorTransform = anchorModule.GetCurrentAnchorTransform();
        Debug.Log("Current Position is: " + anchorTransform.position);
        onAnchorLocationFound?.Invoke();
        debugger.SetIndicatorState("anchor", "ok", "Spatial Anchor Found");
    }

    private void onCreateAnchorSucceeded()
    {
        Debug.Log("Successfully created anchor with ID: "+ anchorModule.currentAzureAnchorID);
        networkUtils.setAnchorID(anchorModule.currentAzureAnchorID);
        anchorTransform = new Pose(gameObject.transform.position, gameObject.transform.rotation);
        onAnchorLocationFound?.Invoke();
        debugger.SetIndicatorState("anchor", "ok", "Spatial Anchor Established");
    }
    #endregion

}
