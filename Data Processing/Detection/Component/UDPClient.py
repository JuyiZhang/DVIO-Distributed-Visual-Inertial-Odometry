import socket
import ipaddress
import struct
import time

from DataManager.Person import Person

UDP_IP = "127.0.0.1"
UDP_PORT = 8848
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_list = []

def send_observed_pose(person_id, coordinate, armature):
    print(len(coordinate[0] + armature[0]))
    send_byte = bytearray(struct.pack("i",2))
    send_byte += bytearray(struct.pack("i",len(person_id)))
    for i in range(0,len(person_id)):
        send_byte += bytearray(struct.pack("i" + "f" * 22, person_id[i], *(coordinate[i] + armature[i]))) # Remember to rotate body angle 90 degree (cw or ccw?) (11 bone * 2 float(yz rotation only) = 22 float * 4 = 88 byte + 4 byte int = 92)
    send_byte += bytearray(struct.pack("f", time.time())) 
    send_data(send_byte)

def send_person(person_list: list[Person]):
    if person_list is None:
        return
    send_byte = bytearray(struct.pack("i",1))
    send_byte += bytearray(struct.pack("i",len(person_list)))
    for person in person_list:
        send_byte += bytearray(struct.pack("i" + "f" * 3, person.id, *(person.coordinate.tolist())))
    send_byte += bytearray(struct.pack("f", time.time())) 
    for device in broadcast_list:
        send_data(send_byte, device)

def send_observed_coordinate(coordinate_list):
    if (coordinate_list is None):
        return
    print(coordinate_list)
    send_byte = bytearray(struct.pack("i",1))
    #send_byte += bytearray(struct.pack("c",b'c'))
    send_byte += bytearray(struct.pack("i",len(coordinate_list)))
    for coordinate in coordinate_list:
        if coordinate[0] is None:
            return
        if len(coordinate[0]) != 4:
            return
        send_byte += bytearray(struct.pack("i" + "f" * 4, int(coordinate[1]), *coordinate[0])) # ID + position (vector3) + rotationY (float)
    send_byte += bytearray(struct.pack("f", time.time())) 
    send_data(send_byte)

def send_all_pose(device_frame: dict):
    send_byte = bytearray(struct.pack("i",0))
    #send_byte += bytearray(struct.pack("c",b'p'))
    send_byte += bytearray(struct.pack("i",len(device_frame)))
    for device_address in device_frame.keys():
        device_frame_data = device_frame[device_address].current_frame.pose.flatten().tolist()
        send_byte += bytearray(struct.pack("i" + "f" * 12, device_address, *device_frame_data)) # (12 float * 4 = 48 + 4 = 52)
    send_byte += bytearray(struct.pack("f", time.time())) 
    send_data(send_byte)
    for device in broadcast_list:
        send_data(send_byte, device)

def send_data(data: bytearray, IP = UDP_IP):
    sock.sendto(data, (IP, UDP_PORT))
