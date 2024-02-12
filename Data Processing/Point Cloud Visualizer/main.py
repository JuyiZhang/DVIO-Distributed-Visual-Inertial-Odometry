
if __name__ == "__main__":
    import Component.Debug as Debug
    import time
    from PyQt5.QtWidgets import QApplication
    import sys
    from Component.BackendUI import app_window  
    
    app = QApplication([])
    window = app_window()
    
    print("\n")
    Debug.Log("Session initiated", True)
    Debug.Log("Session begins at: " + time.strftime("%Y/%m/%d %H:%M:%S, UTC %z"), True)
    Debug.Log("Debug mode is now set to " + str(Debug.debug_mode), True)
    Debug.Log("Begin Debugging...")
    print("\n")
    Debug.Log("Starting MuTA Backend...")
    
    sys.exit(app.exec())