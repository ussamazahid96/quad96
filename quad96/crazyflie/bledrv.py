import time
import queue
import struct
import bluepy as b
from bluepy.btle import DefaultDelegate

from cflib.crtp.crtpstack import CRTPPort
from cflib.crtp.crtpstack import CRTPPacket
from cflib.crtp.crtpdriver import CRTPDriver


__author__ = "Ussama Zahid"
__email__ = "ussamazahid96@gmail.com"

# this file contains the bluetooth driver defination

SERVICE_UUID = "000002011c7f4f9e947b43b7c00a9a08"
CRTP_UUID = "000002021c7f4f9e947b43b7c00a9a08"
CRTP_UP_UUID = "00000203-1C7F-4F9E-947B-43B7C00A9A08"
CRTP_DOWN_UUID = "00000204-1C7F-4F9E-947B-43B7C00A9A08"

# control byte of the incomming CRTP packet bluetooth
class Control_Byte:
    def __init__(self, start=False, pid=-1, length=-1, raw_packet=None):
        if raw_packet is not None:
            self.start = (raw_packet[0] & 0x80) != 0
            self.pid = (raw_packet[0] >> 5) & 0x03
            self.length = (raw_packet[0] & 0x1F)
        else:
            self.start = start
            self.pid = pid
            self.length = length

    def to_byte(self):
        ret = 0x80 if self.start else 0x00
        ret = ret | ( (self.pid&0x03) << 5 ) | ( (self.length-1)&0x1f )        
        return struct.pack('<B', ret)

# this class is used to handle the notification on the changes of the monitored value
class MyDelegate(DefaultDelegate):
    def __init__(self, in_queue):
        DefaultDelegate.__init__(self)
        self.in_queue = in_queue
        self.tempByteArray = bytearray()
        self.temp_pid = -1
        self.temp_length = -1

    def handleNotification(self, cHandle, data):
        raw_packet = bytearray(data)
        header = Control_Byte(raw_packet=raw_packet)
        return_packet = None
        
        if (header.start):
            if header.length < 20:
                self.tempByteArray.clear()
                self.tempByteArray[0:header.length-1] = raw_packet[1:]
                return_packet = CRTPPacket(header=self.tempByteArray[0], data=self.tempByteArray[1:])
                self.in_queue.put(return_packet)
            else:
                self.tempByteArray[0:header.length-1] = raw_packet[1:]
                self.temp_pid = header.pid
                self.temp_length = header.length
        else:
            if header.pid == self.temp_pid:
                self.tempByteArray[19:header.length-1] = raw_packet[1:]
                return_packet = CRTPPacket(header=self.tempByteArray[0], data=self.tempByteArray[1:])
                self.in_queue.put(return_packet)
            else:
                self.temp_pid = -1
                self.temp_length = 0


# the main bluetooth driver class

class BLEDriver(CRTPDriver):
    def __init__(self, address=None):
        CRTPDriver.__init__(self)
        self.address_give = address
        self.needs_resending = True
        self.in_queue = queue.Queue()
        
        self.pid = 0
        self.service = None
        self.crtp = None
        self.crtp_up = None
        self.crtp_down = None


        self.scanner = b.btle.Scanner()
        self.peripheral = b.btle.Peripheral()
        self.peripheral.setDelegate(MyDelegate(self.in_queue))
        

    def enable_notify(self,  notify):
        setup_data = b"\x01\x00"
        notify_handle = notify.getHandle() + 1
        self.peripheral.writeCharacteristic(notify_handle, setup_data, withResponse=True)


    def connect(self, uri, link_quality_callback=None, link_error_callback=None):
        """Connect the driver to a specified URI

        @param uri Uri of the link to open
        @param link_quality_callback Callback to report link quality in percent
        @param link_error_callback Callback to report errors (will result in
               disconnection)
        """
        self.peripheral.connect(uri, addrType=b.btle.ADDR_TYPE_RANDOM)
        self.service = self.peripheral.getServiceByUUID(b.btle.UUID(SERVICE_UUID))
        self.crtp = self.service.getCharacteristics(b.btle.UUID(CRTP_UUID))[0]
        self.crtp_up = self.service.getCharacteristics(b.btle.UUID(CRTP_UP_UUID))[0]
        self.crtp_down = self.service.getCharacteristics(b.btle.UUID(CRTP_DOWN_UUID))[0]
        self.enable_notify(self.crtp_down)

    def send_packet(self, pk):
        """Send a CRTP packet"""
        if (len(pk.data) <=20):
            header = struct.pack('<B', pk.get_header())
            dataOut = header + pk.data
            self.crtp.write(dataOut)
        else:
            self.send_split_packet(pk)


    def send_split_packet(self, pk2):
        header = struct.pack('<B', pk.get_header())
        
        first_packet = bytearray()
        first_packet = Control_Byte(True, self.pid, len(header + pk.data)).to_byte()
        first_packet += header
        first_packet += pk.data[0:18]
        self.crtp_up.write(bytes(first_packet))
        time.sleep(0.05)

        second_packet = bytearray()
        second_packet = Control_Byte(False, self.pid, 0).to_byte()
        second_packet += pk.data[19:]
        self.crtp_up.write(bytes(second_packet))
        time.sleep(0.05)

        self.pid = (self.pid+1)%4 


    def receive_packet(self, wait=0):
        """Receive a CRTP packet.

        @param wait The time to wait for a packet in second. -1 means forever

        @return One CRTP packet or None if no packet has been received.
        """
        try:
            if wait == 0:
                pk = self.in_queue.get(False)
            elif wait < 0:
                pk = self.in_queue.get(True)
            else:
                pk = self.in_queue.get(True, wait)
        except queue.Empty:
            return None
        
        return pk

    def get_status(self):
        """
        Return a status string from the interface.
        """
        return 'No information available'

    def get_name(self):
        """
        Return a human readable name of the interface.
        """
        return "/dev/hci0"


    def scan_interface(self, address=None):
        """
        Scan interface for available Crazyflie quadcopters and return a list
        with them.
        """
        cf = []
        print("Running Bluetooth Scan for 10 sec...")
        devices = self.scanner.scan()
        for dev in devices:
            header = dev.getScanData()[0]
            if header[-1] == "Crazyflie":
                cf += [dev.addr]
                print("Crazyflie found with MAC addr = {}".format(dev.addr))
        return [cf]

    def enum(self):
        """Enumerate, and return a list, of the available link URI on this
        system
        """
        pass

    def get_help(self):
        """return the help message on how to form the URI for this driver
        None means no help
        """
        return "Use MAC address of the Crazyflie to connect."

    def close(self):
        """Close the link"""
        while not self.in_queue.empty():
            self.in_queue.get()
        self.peripheral.disconnect()
        self.service = None
        self.crtp = None
        self.crtp_up = None
        self.crtp_down = None





