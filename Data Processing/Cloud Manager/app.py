from flask import Flask, request, g, current_app, render_template
from flask_pymongo import PyMongo
from werkzeug.local import LocalProxy
import json
import time
import os

app = Flask(__name__)

@app.route("/")
def home():
    anchorData = json.loads(queryAnchor())
    return render_template("index.html", anchor_name = "Found" if anchorData["result"] == "success" else "Not Found")

@app.route("/add_anchor", methods=['POST'])
def addAnchor():
    
    if request.method == 'POST':
        anchorid = request.form.get("anchorID")
        creator = request.form.get("creator")
        ctime = str(int(time.time_ns()/1000000))
    
    result = {
        "result": "success",
        "id": anchorid,
        "creator": creator,
        "ctime": ctime 
    }
    
    json_result = json.dumps(result)
    with open("anchor_data.json","w+") as outfile:
        outfile.write(json_result)
        
    return json_result

@app.route("/remove_anchor")
def removeAnchor():
    if os.path.exists("anchor_data.json"):
        os.remove("anchor_data.json")
        return render_template("action.html", result = "Success")
    else:
        return render_template("action.html", result = "No Anchor Found")

@app.route("/query_anchor")
def queryAnchor():
    
    if not(os.path.exists("anchor_data.json")):
        
        anchor_result = {
            "result": "failed",
            "reason": "0",
            "reason_string": "Anchor not found"
        }
        
        return json.dumps(anchor_result)
    
    anchor_file = open("anchor_data.json")
    
    anchorData = json.load(anchor_file)
    
    return anchorData

@app.route("/add_host", methods=["POST"])
def addHost():
    
    request_data = request.get_json()
    
    print(request_data)
    
    if request.method == "POST":
        if request_data["op"] == "delete":
            hostname = ""
            if (os.path.exists("host_data.json")):
                os.remove("host_data.json")
            return json.dumps({"result": "success"})
        else:
            hostname = request_data["hostname"]
    
    result = {
        "result": "success",
        "hostname": hostname,
    }
    json_result = json.dumps(result)
    with open("host_data.json","w+") as outfile:
        outfile.write(json_result)
        
    return json_result

@app.route("/query_host")
def queryHost():
    
    if not(os.path.exists("host_data.json")):
        
        """host_result = {
            "result": "failed",
            "reason": "1",
            "reason_string": "Host Not Online"
        }
        
        return json.dumps(host_result)"""
        return "Offline"
    
    host_file = open("host_data.json")
    
    host_json = json.load(host_file)
    
    return host_json["hostname"]

@app.route("/version")
def version():
    return "0.0.4:110820231604"

@app.route("/qr_test")
def qr_test():
    qr_result = {
        "title": "Hello World",
        "caption": "This field is used to test out the content included in the qrcode and shall be replaced in the future with actual test. For right now, just sit back and enjoy the content. Good luck! ;)"
    }
    return(json.dumps(qr_result))
    
    
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = PyMongo(current_app).db

db = LocalProxy(get_db)