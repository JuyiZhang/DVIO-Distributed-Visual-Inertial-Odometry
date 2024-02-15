import ipaddress
from multiprocessing.managers import DictProxy
import socket
import struct
import traceback
import os

import Component.Debug as Debug

import Component.NetworkManager as NetworkManager
import DataManager.FrameFile as FrameFile

from multiprocessing import Process


    
server_host = ''
server_port = 9090


def close_connection():
    NetworkManager.del_host()

def listen_for_connection_async(session_folder, device_timestamp: DictProxy, direct_feed = False):
    NetworkManager.add_host(socket.gethostbyname(socket.gethostname()))
    socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_connection.bind((server_host, server_port))
        Debug.Log('Server bind to port ' + str(server_port))
    except socket.error as msg:
        Debug.Log('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
        return
    socket_connection.listen(10)
    Debug.Log('Start listening...')
    socket_connection.settimeout(3.0)
    while True:
        try:
            conn, addr = socket_connection.accept() # Blocking, wait for incoming connection
            Debug.Log('Connected with ' + addr[0] + ':' + str(addr[1]))
            Process(target=listen_for_data, args=(conn, addr, session_folder, device_timestamp, direct_feed)).start()
        except KeyboardInterrupt:
            return
        except Exception:
            continue
        
def listen_for_data(conn, addr, session_folder: str, device_timestamp_dict: DictProxy, direct_feed: bool):
    Debug.Log('Connected with ' + addr[0] + ':' + str(addr[1]))
    device_address = int(ipaddress.ip_address(addr[0]))
    if not(os.path.exists(session_folder + "/" + str(device_address))):
        os.mkdir(session_folder + "/" + str(device_address))
    
    final_data = bytearray(0)
    current_header = ""
    current_data_length = 0

    while True:
        # Receiving and transmitting from client
        try:
            data = conn.recv(512*512*4)
            
            #while len(self.transmit_queue) > 0:
            #    conn.send(self.transmit_queue.pop(0))
            
            if len(data)==0:
                continue
            if current_header != "": # Means there are data that is not yet received
                final_data += bytearray(data)
                if current_data_length + 1 > len(final_data):
                    continue
                else:
                    if current_header == "c":
                        current_header = ""
                        processData = final_data
                        if len(final_data) > current_data_length + 1:
                            data = final_data[current_data_length:len(processData)]
                            processData = final_data[0:current_data_length]
                            FrameFile.data_process(processData, device_address, device_timestamp_dict, session_folder, direct_feed)
                            #data_post_process(finalData, save_folder)
                            current_data_length = 0
                            final_data.clear() 
                        else:
                            processData = final_data
                            print("f")
                            FrameFile.data_process(processData, device_address, device_timestamp_dict, session_folder, direct_feed)
                            #data_post_process(finalData, save_folder)
                            current_data_length = 0
                            final_data.clear()
                            continue
                    
            try:
                header = data[0:1].decode('utf-8')
                Debug.Log('-------------------------- Header: ' + header + ' --------------------------', category="Conn")
            except:
                Debug.Log('Header is not UTF8', category="Conn")
                continue

            if header == 'c':
                current_data_length = 0
                final_data.clear()
                # Retrieve Depth Sensor Data
                depth_length = struct.unpack(">i", data[9:13])[0]
                ab_length = struct.unpack(">i", data[13:17])[0]
                pointcloud_length = struct.unpack(">i", data[17:21])[0]
                if (depth_length != 184320 or ab_length != 92160 or pointcloud_length <= 0):
                    continue
                final_data += bytearray(data)
                current_header = 'c'
                current_data_length = depth_length + ab_length + pointcloud_length + 21
                Debug.Log("Length of expected data " + str(current_data_length), category="Conn")
                if len(data) < current_data_length: # Insufficient data, continue receive
                    continue
                else:
                    current_header = ""
                    Debug.Log("Adding new frame", "Frame Important")
                    FrameFile.data_process(data, addr[0], device_timestamp_dict, session_folder, direct_feed)
                    current_data_length = 0
                    final_data.clear()
        except ConnectionResetError:
            if addr[0] in device_timestamp_dict.keys():
                device_timestamp_dict.pop(addr[0])
            traceback.print_exc()
            return
        except KeyboardInterrupt:
            return
        except:
            traceback.print_exc()
            continue
            