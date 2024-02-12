using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;

public class IPManager
{
    public static IPAddress GetIP()
    {
        foreach (NetworkInterface netInterface in NetworkInterface.GetAllNetworkInterfaces())
        {
            if (netInterface.NetworkInterfaceType == NetworkInterfaceType.Wireless80211 && netInterface.OperationalStatus == OperationalStatus.Up)
            {
                foreach (UnicastIPAddressInformation ip in netInterface.GetIPProperties().UnicastAddresses)
                {
                    if (ip.Address.AddressFamily == AddressFamily.InterNetwork)
                    {
                       return ip.Address;
                    }
                }
            }
        }
        return IPAddress.None;
    }
}

