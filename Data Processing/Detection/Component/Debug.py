import time

global prev_time
prev_time = time.time()
debug_mode = True
filter_category = ["Conn", "Frame"]

#fp = open("log/Session_"+str(int(time.time()))+".log","a")

def Log(log_content, force_print = False, category = "0"):
    
    if (debug_mode or force_print):
        if not(category in filter_category):
            if (category == "Performance"):
                global prev_time
                print("[" + time.strftime("%H:%M:%S:") + str(int(time.time() * 1000 % 1000)) + "] {" + category + "} " + log_content + " with time " + str(time.time() - prev_time))
                prev_time = time.time()
            else:
                print("[" + time.strftime("%H:%M:%S:") + str(int(time.time() * 1000 % 1000)) + "] {" + category + "} " + log_content)
            #fp.write("[" + time.strftime("%H:%M:%S:") +  + "] {" + category + "} " + log_content + "\n")
            
        
