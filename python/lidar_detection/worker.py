# SPDX-License-Identifier: Apache-2.0
"""Persistent binary worker. stdout is reserved for the versioned IPC protocol."""
import argparse
import contextlib
import struct
import sys

REQUEST=struct.Struct('<4sQIf')
RESPONSE=struct.Struct('<4sQIfI')
MAX_POINTS=4_000_000

def read_exact(stream,count):
    result=bytearray(count)
    read_into(stream,memoryview(result))
    return result

def read_into(stream,target):
    """Fill an existing receive buffer, including short pipe reads."""
    offset=0
    while offset<len(target):
        size=stream.readinto(target[offset:])
        if not size:raise EOFError('Detector request pipe closed')
        offset+=size

def reply(stream,magic,revision,boxes,latency,message=''):
    message=message.encode('utf-8')[:16384]
    def write_all(data):
        data=memoryview(data)
        while len(data):
            size=stream.write(data)
            if not size:raise BrokenPipeError('Detector reply pipe closed')
            data=data[size:]
    write_all(RESPONSE.pack(magic,revision,len(boxes),latency,len(message)))
    write_all(message)
    if len(boxes):write_all(memoryview(boxes.astype('<f4',copy=False)).cast('B'))
    stream.flush()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--provider',choices=['auto','cpu'],default='auto')
    args=parser.parse_args()
    source=sys.stdin.buffer;sink=sys.stdout.buffer
    # Some optional native libraries print diagnostic messages to stdout.
    # Redirect fd 1 too; retain a private duplicate for binary replies.
    import os
    sink=os.fdopen(os.dup(sink.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),1)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            import numpy as np
            from lidar_detection.centerpoint import CenterPoint
            detector=CenterPoint(args.model_dir,args.provider)
            reply(sink,b'LDR1',0,[],0,detector.provider)
        except Exception as error:
            reply(sink,b'LDE1',0,[],0,str(error));return 1
        receive=bytearray()
        while True:
            try:header=read_exact(source,REQUEST.size)
            except EOFError:return 0
            magic,revision,count,threshold=REQUEST.unpack(header)
            if magic!=b'LDQ1' or count>MAX_POINTS or not .01<=threshold<=.99:
                reply(sink,b'LDE1',revision,[],0,'Invalid detector request');return 1
            try:
                needed=count*12
                if len(receive)<needed:receive=bytearray(needed)
                read_into(source,memoryview(receive)[:needed])
                points=np.frombuffer(receive,dtype='<f4',count=count*3).reshape(-1,3)
                boxes,stats=detector.infer(points,threshold)
                message='pre %.1f / net %.1f / post %.1f ms; %d pillars; %d truncated'%(
                    stats.get('preprocess_ms',0),stats.get('inference_ms',0),stats.get('postprocess_ms',0),stats['pillars'],stats['truncated_pillars'])
                reply(sink,b'LDR1',revision,boxes,stats['latency_ms'],message)
            except Exception as error:
                reply(sink,b'LDE1',revision,[],0,str(error))

if __name__=='__main__':raise SystemExit(main())
