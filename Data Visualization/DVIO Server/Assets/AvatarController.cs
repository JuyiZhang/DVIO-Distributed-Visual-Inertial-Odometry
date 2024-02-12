using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AvatarController : MonoBehaviour
{
    // Start is called before the first frame update
    [SerializeField]
    private Transform body;
    [SerializeField]
    private Transform spine;
    [SerializeField]
    private Transform neck;
    [SerializeField]
    private Transform leftShoulder;
    [SerializeField]
    private Transform rightShoulder;
    [SerializeField]
    private Transform leftElbow;
    [SerializeField]
    private Transform rightElbow;
    [SerializeField]
    private Transform leftHip;
    [SerializeField]
    private Transform rightHip;
    [SerializeField]
    private Transform leftKnee;
    [SerializeField]
    private Transform rightKnee;

    private Transform[] boneArray;
    void Start()
    {
        Transform[] bone = { body, spine, neck, leftShoulder, rightShoulder, leftElbow, rightElbow, leftHip, rightHip, leftKnee, rightKnee };
        boneArray = bone;
    }

    public void setAllBone(Quaternion[] rotations)
    {
        for (int i=0;i<boneArray.Length;i++)
        {
            boneArray[i].transform.rotation = rotations[i];
        }
    }

}
