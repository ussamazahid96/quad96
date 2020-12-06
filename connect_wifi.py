# sample script to connect ultra96 to wifi

from pynq.lib import Wifi

port = Wifi()

ssid = str(input("Enter ssid: "))
pwd = str(input("Enter password: "))

port.connect(ssid, pwd)

