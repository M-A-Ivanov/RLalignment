"""
Created on Mon Sep 29 15:51:22 2014
@author: Helder Marchetto
*** Diamond Instruction ***
Connnecting to uView is only possible from Python and
not from Jython (the numpy library is unavailable in Jython).
1) Open a Terminal
2) Type in "python":
[i06user@i06-ws001 Desktop]$ python
3) Type in the following lines:
import sys
import os
sys.path.append(os.path.abspath("/dls_sw/i06/scripts/maccherozzi/TCP_Elmitec"))
import UviewConnect
4) Start the connection to Uview with:
uvc = UviewConnect.oUview()
Notice that if the Uview PC IP or Port changes, you have to connect this way:
uvc = UviewConnect.oUview(ip= '172.23.106.79', port=5570)
Substituting the appropriate IP address and Port.
Basic instructions:
See all available functions with:
>>> dir(uvc)
['TCPBlockingReceive', 'UviewConnected', '__class__', '__delattr__', '__dict__',
'__doc__', '__enter__', '__exit__', '__format__', '__getattribute__', '__hash__',
'__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
'__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
'acquireSingleImg', 'aip', 'connect', 'disconnect', 'exportImage', 'getAcqState',
'getAvr', 'getCameraSize', 'getExposureTime', 'getImage', 'getMarkerInfo',
'getNrActiveMarkers', 'getROI', 'getTcp', 'ip', 'lastTime', 'port', 's',
'setAcqState', 'setAvr', 'setExposureTime', 'setIP', 'setPort', 'setTcp',
'testConnect']
Or use help(uvc) and get a long list:
Help on oUview in module UviewConnect object:
class oUview(__builtin__.object)
 |  Methods defined here:
 |
 |  TCPBlockingReceive(self)
 |
 |  __enter__(self)
 |
 |  __exit__(self, type, value, traceback)
 |
 |  __init__(self, ip='172.23.106.79', port=5570, directConnect=True)
 |
 |  acquireSingleImg(self, id=-1)
 |
 |  aip(self)
 |
 |  connect(self)
 |
 |  disconnect(self)
 |
 |  exportImage(self, fileName, imgFormat='0', imgContents='0')
 |
 |  getAcqState(self)
 |
 |  getAvr(self)
 |
 |  getCameraSize(self)
 |
 |  getExposureTime(self)
 |
 |  getImage(self)
...
Get the average status:
>>> uvc.getAvr()
1
Where "1" is the number in the drop down menu in Uview (0:no average;1:sliding average;2:average 2;...)
Set the average to no average (nr 0):
>>> uvc.setAvr(0)
'0'
Get an image with:
>>> img = uvc.getImage()
>>> img.shape
(512, 512)
>>> type(img)
<type 'numpy.ndarray'>
"""
import socket
import time
import numpy as np
import struct
import sys


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class oUview(object):
    UviewConnected = False

    def __enter__(self):
        return self

    def __init__(self, ip='172.23.106.79', port=5570, directConnect=True):
        if type(ip) != str:
            print
            'Uview_Host must be a string. Using localhost instead.'
            self.ip = '172.23.106.79'
        else:
            self.ip = ip
        if type(port) != int:
            print
            'Uview_Port must be an integer. Using 5570.'
            self.port = 5570
        else:
            self.port = port
        self.lastTime = time.time()
        if directConnect:
            print
            'Connect with ip=' + str(self.ip) + ', port=' + str(self.port)
            self.connect()

    def __exit__(self, type, value, traceback):
        # print "Destroying", self
        if self.UviewConnected:
            # print 'Exit without closing connections... close connections'
            self.s.send('clo'.encode())
            self.s.close()
            self.UviewConnected = False

    def connect(self):
        if self.UviewConnected:
            print
            'Already connected ... exit "connect" method'
            return None
        else:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.ip, self.port))
            self.UviewConnected = True
            # Start string communication
            TCPString = 'asc'
            self.s.send(TCPString.encode())
            data = self.TCPBlockingReceive()

    def testConnect(self):
        if self.UviewConnected:
            print
            'Already connected ... exit "connect" method'
            return True
        else:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.settimeout(5)  # 5 seconds timeout
            t = time.time()
            try:
                self.s.connect((self.ip, self.port))
                print
                'connected in ' + str(time.time() - t) + ' sec'
                self.s.settimeout(None)
                self.s.close()
                return True
            except:
                print
                'connection not possible'
                print
                'please check: that Uview is running'
                print
                '              the IP address is correct'
                print
                '              the PORT is the same activated in Uview'
                return False
            else:
                return False

    def setIP(self, IP):
        if self.UviewConnected:
            print
            'Already connected ... exit "connect" method'
            return
        else:
            if type(IP) != str:
                print
                'The IP has to be a string. Please use this synthax:'
                print
                "object.setIP('192.168.1.0')"
                print
                "or"
                print
                "object.setIP('localhost')"
                return
            self.ip = str(IP)

    def setPort(self, port):
        if self.UviewConnected:
            print
            'Already connected ... exit "connect" method'
            return
        else:
            if type(port) != int:
                print
                'The port has to be a number. Please use this synthax:'
                print
                "object.setPort(5570)"
                return
            self.port = str(port)

    def disconnect(self):
        if self.UviewConnected:
            self.s.send('clo'.encode())
            self.s.close()
            self.UviewConnected = False
            print
            "Disconnected!"

    def getImage(self):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            TCPString = 'ida 0 0'
            self.s.send(TCPString.encode())
            header = ''
            for i in range(19):
                header += self.s.recv(1)
            arr = header.split()
            if len(arr) != 3:
                print
                'Wrong header. Exit'
                return
            xs = int(arr[1])
            ys = int(arr[2])
            img = np.zeros((xs, ys), dtype=np.uint16)  # must be 16 bit
            if sys.version_info >= (2, 7):
                for i in range(ys):
                    img[:, i] = struct.unpack('{}H'.format(xs), self.s.recv(xs * 2))
            else:
                for i in range(ys):
                    img[:, i] = struct.unpack('{0}H'.format(xs), self.s.recv(xs * 2))
            return img

    def exportImage(self, fileName, imgFormat='0', imgContents='0'):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            TCPString = 'exp ' + str(imgFormat) + ', ' + str(imgContents) + ', ' + str(fileName)
            self.s.send(TCPString.encode())
            data = self.TCPBlockingReceive()
            return len(data) == 0

    def setAvr(self, avr='0'):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            if not is_number(avr):
                return
            numAvr = int(avr)
            if (numAvr >= 0) and (numAvr <= 99):
                retVal = self.setTcp('avr'.encode(), str(numAvr))
                return retVal

    def getAvr(self):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            TCPString = 'avr'
            self.s.send(TCPString.encode())
            data = self.TCPBlockingReceive()
            if is_number(data):
                return int(data)
            else:
                return 0

    def acquireSingleImg(self, id=-1):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            TCPString = 'asi ' + str(id)
            self.s.send(TCPString.encode())
            return self.TCPBlockingReceive()

    def setAcqState(self, acqState=-1):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            if (acqState == -1) or (acqState == '-1'):
                acqState = self.aip()
            acqState = str(acqState)
            if (acqState != '0') or (acqState != '1'):
                return
            TCPString = 'aip ' + str(acqState)
            self.s.send(TCPString.encode())
            return self.TCPBlockingReceive()

    def getAcqState(self):
        return self.aip()

    def aip(self):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            TCPString = 'aip'
            self.s.send(TCPString)
            return self.TCPBlockingReceive() == '1'

    def getROI(self):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            xmi = self.getTcp('xmi', False, True)
            xma = self.getTcp('xma', False, True)
            ymi = self.getTcp('ymi', False, True)
            yma = self.getTcp('yma', False, True)
            return [xmi, ymi, xma, yma]

    def getCameraSize(self):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            gcs = self.getTcp('gcs', False, False, True)
            spl = gcs.split()
            if len(spl) == 2:
                if is_number(spl[0]) and is_number(spl[1]):
                    return [int(spl[0]), int(spl[1])]
                else:
                    print
                    'Uview image size format error'
                    return [-1, -1]
            else:
                print
                'Uview image size format error'
                return [-1, -1]

    def getExposureTime(self):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            ext = self.getTcp('ext', True, False, False)
            return ext

    def setExposureTime(self, ext):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            self.setTcp('ext', ext)
            return

    def getNrActiveMarkers(self):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            print
            'call nMarkersStr'
            nMarkersStr = self.getTcp('mar -1', False, False, True)
            print
            'nMarkersStr returns = ', nMarkersStr
            nMarkers = nMarkersStr.split()
            if len(nMarkers) <= 1:
                return 0
            if not is_number(nMarkers[0]):
                return 0
            nm = int(nMarkers[0])
            del nMarkers[0]
            markers = []
            for i in nMarkers: markers.append(int(i))
            return [nm, markers]

    def getMarkerInfo(self, marker):
        if not self.UviewConnected:
            print
            'Please connect first'
            return None
        else:
            if not is_number(marker):
                print
                'Marker value must be a valid number'
                return
            markerInfo = self.setTcp('mar', str(marker))
            splitMarker = markerInfo.split()
            if len(splitMarker) != 7:
                return 0
            typeNr = int(splitMarker[2])
            if typeNr == 0:
                myType = 'line'
            elif typeNr == 1:
                myType = 'horizline'
            elif typeNr == 2:
                myType = 'vertline'
            elif typeNr == 5:
                myType = 'circle'
            elif typeNr == 9:
                myType = 'text'
            elif typeNr == 10:
                myType = 'cross'
            else:
                myType = 'unknown'
            return {'marker': marker,
                    'imgNr': int(splitMarker[1]),
                    'type': myType,
                    'typeNr': typeNr,
                    'pos': [int(splitMarker[3]), int(splitMarker[4]), int(splitMarker[5]), int(splitMarker[6])]}

    def getTcp(self, TCPString, isFlt=True, isInt=False, asIs=False):
        self.s.send(TCPString)
        retStr = self.TCPBlockingReceive()
        if asIs:
            # print 'is asIs = ', TCPString, retStr
            return retStr
        elif isInt:
            if is_number(retStr):
                return int(retStr)
            else:
                return 0
        elif isFlt:
            if is_number(retStr):
                return float(retStr)
            else:
                return 0.0
        else:
            return retStr

    def setTcp(self, TCPString, Value):
        TCPString = TCPString.strip() + ' ' + Value.strip()
        self.s.send(TCPString)
        return self.TCPBlockingReceive()

    def TCPBlockingReceive(self):
        Bytereceived = '0'
        szData = ''
        while ord(Bytereceived) != 0:
            ReceivedLength = 0
            while ReceivedLength == 0:
                Bytereceived = self.s.recv(1)
                # print 'Bytereceived=',Bytereceived,'ord(Bytereceived)=',ord(Bytereceived)
                ReceivedLength = len(Bytereceived)
            if ord(Bytereceived) != 0:
                szData = szData + Bytereceived
        return szData.decode()