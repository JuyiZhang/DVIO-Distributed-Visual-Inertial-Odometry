import requests
import Component.Debug as Debug

url = "https://mutaw.azurewebsites.net/"
    
# Adding host to server when online
def add_host(hostip):
    
    Debug.Log("Adding Host IP: " + hostip)
    request_url = url + "add_host"
    request_content = {
        "op": "addm", # Add or Modify Host
        "hostname": hostip
    }
    return requests.post(request_url,json=request_content)

# Deleting host to server when offline
def del_host():
    
    request_url = url + "add_host"
    request_content = {
        "op": "delete", # Delete Host
    }
    return requests.post(request_url, json=request_content)

