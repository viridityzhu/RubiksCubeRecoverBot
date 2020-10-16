# coding=utf-8
import serial
import time


def send_arduino(com, txt):
    try:
        x = serial.Serial(com, 9600)
        if x.isOpen() > 0:
            print("open com")
            time.sleep(2)
            txt = txt[:-6]
            x.write(txt.encode())
            print('write:')
            print(txt.encode())
            x.close()
    except Exception as e:
        print('Cannot open com.')
        print(e)
