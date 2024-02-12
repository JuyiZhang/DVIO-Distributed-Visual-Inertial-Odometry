using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class TCPServer : MonoBehaviour
{

    private TcpListener tcpListener;
    private Thread tcpListenerThread;
    private TcpClient connectedTcpClient;

    [SerializeField]
    private int port = 9090;
    // Start is called before the first frame update
    void Start()
    {
        tcpListenerThread = new Thread(new ThreadStart(ListenForIncomingConnection));
        tcpListenerThread.IsBackground = true;
        tcpListenerThread.Start();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void ListenForIncomingConnection()
    {
        try
        {
            IPAddress ipAddress = IPManager.GetIP();
            tcpListener = new TcpListener(ipAddress, port);
            tcpListener.Start();
            Debug.Log("Server Listening at " + ipAddress.ToString());
            while(true)
            {
                TcpClient client = tcpListener.AcceptTcpClient();
                Thread clientThread = new Thread(new ThreadStart(() => acceptIncomingData(client)));
                clientThread.IsBackground = true;
                clientThread.Start();
            }

        } catch
        {

        }
    }

    private void acceptIncomingData(TcpClient client)
    {
        Debug.Log("Server connected to client at " + (client.Client.RemoteEndPoint as IPEndPoint).Address);
        NetworkStream stream = client.GetStream();
        int i;
        byte[] data = new byte[1024];
        while (true)
        {
            i = stream.Read(data, 0, data.Length);
            if (i != 0)
            {
                string type = Encoding.UTF8.GetString(data,0,1);
                if (type == "c")
                {
                    long timestamp = BitConverter.ToInt64(data, 1);
                    int depthMapLength = BitConverter.ToInt32(data, 9);
                    int abImageLength = BitConverter.ToInt32(data, 13);
                    int pointCloudLength = BitConverter.ToInt32(data, 17);
                    Debug.Log("Receive data at timestamp " + timestamp);
                }
            }
        }

    }
}
