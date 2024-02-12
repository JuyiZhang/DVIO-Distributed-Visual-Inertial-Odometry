using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Debugger: MonoBehaviour
{
    private List<string> debugMessageHistory = new List<string>();

    [SerializeField]
    private TMPro.TextMeshProUGUI debugText;

    [SerializeField]
    private GameObject tcpIndicator;

    [SerializeField]
    private GameObject anchorIndicator;

    [SerializeField]
    private GameObject sendIndicator;

    [SerializeField]
    private TMPro.TextMeshProUGUI debugModeBtn;

    [SerializeField]
    private Michsky.UI.Shift.SwitchManager continuousSendSwitch;

    [SerializeField]
    private GameObject configCanvas;

    private void Start()
    {
        debugText.text = "Debugging started";
        Application.logMessageReceived += AddLogMessage;
    }

    public void AddLogMessage(string message, string stackTrace, LogType type)
    {
        AddDebugMessage(message);
    }

    public void AddDebugMessage(string debugMsg)
    {
        debugMessageHistory.Insert(0, debugMsg);
        debugText.text = string.Join(System.Environment.NewLine, debugMessageHistory.ToArray());
    }

    public void ToggleDebugMode()
    {
        debugText.enabled = !debugText.enabled;
    }

    public void SetIndicatorState(string indicator, string state, string state_text)
    {
        GameObject indicator_object = new GameObject();
        if (indicator == "anchor")
        {
            indicator_object = anchorIndicator;
        } else if (indicator == "tcp")
        {
            indicator_object = tcpIndicator;
        } else if (indicator == "send")
        {
            indicator_object = sendIndicator;
        }
        TMPro.TextMeshProUGUI indicator_text = indicator_object.transform.GetChild(1).GetComponent<TMPro.TextMeshProUGUI>();
        Image indicator_glow = indicator_object.transform.GetChild(0).GetComponent<Image>();
        Image indicator_image = indicator_object.transform.GetChild(2).GetComponent<Image>();
        Animator indicator_animator = indicator_object.GetComponent<Animator>();
        Color apply_t = new Color();
        Color apply = new Color();
        if (state == "ip") // In Progress
        {
            indicator_animator.SetBool("Loop", true);
            apply_t = new Color(0.386f, 0.772f, 0.99f, 0.1f);
            apply = new Color(0.386f, 0.772f, 0.99f);
            
        }
        else if (state == "error")
        {
            apply_t = new Color(1.0f, 0.4f, 0.4f, 0.1f);
            apply = new Color(1.0f, 0.4f, 0.4f);
            indicator_animator.SetBool("Loop", false);
        }
        else if (state == "ok")
        {
            apply_t = new Color(0.4f, 1f, 0.4f, 0.1f);
            apply = new Color(0.4f, 1f, 0.4f);
            indicator_animator.SetBool("Loop", true);
        } else
        {
            apply_t = new Color(0.4f, 1f, 0.4f, 0.1f);
            apply = new Color(0.4f, 1f, 0.4f);
            indicator_animator.SetBool("Loop", false);
        }
        indicator_image.material.color = apply;
        indicator_text.faceColor = apply;
        indicator_text.text = state_text;
    }

    public void toggleContinuousSendSwitch()
    {
        continuousSendSwitch.AnimateSwitch();
    }

    public void OpenConfigWindow()
    {
        configCanvas.SetActive(true);
    }

    public void CloseConfigWindow()
    {
        configCanvas.SetActive(false);
    }

}
