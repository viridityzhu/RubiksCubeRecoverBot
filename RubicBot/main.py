import start_server
from threading import Thread
background_thread = Thread(target=start_server.start, args=(8080, 20, 2))
background_thread.start()
# 在port 8080打开server, 最大20步, timeout 2s
import client_gui
