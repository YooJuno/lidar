"""Controls located from actual ImGui rectangles, on private Xvfb only."""
import re
import time


class Controls:
    def __init__(self,x11,window,log):self.x11=x11;self.window=window;self.log=log
    def rect(self,name,timeout=8,after=0):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            rows=re.findall(r'^UI '+re.escape(name)+r' ([-.\d]+) ([-.\d]+) ([-.\d]+) ([-.\d]+)$',self.log.read_text(errors='replace')[after:],re.M)
            if rows:return tuple(map(float,rows[-1]))
            time.sleep(.05)
        raise AssertionError('Control not rendered: '+name)
    def click(self,name):
        x,y,r,b=self.rect(name);self.x11.click(self.window,int((x+r)*.5),int((y+b)*.5));time.sleep(.1)
    def event(self,key,down):
        x=self.x11;code=x.x.XKeysymToKeycode(x.display,x.x.XStringToKeysym(key.encode()))
        x.xt.XTestFakeKeyEvent(x.display,code,down,0)
    def chord(self,key):
        self.event('Control_L',True);self.event(key,True);self.event(key,False);self.event('Control_L',False)
        self.x11.x.XSync(self.x11.display,False);time.sleep(.2)
    def enter(self,name,value,axis=None,ctrl_click=False):
        x,y,r,b=self.rect(name);at=(x+r)*.5 if axis is None else x+(r-x)*(axis+.5)/3
        if ctrl_click:self.event('Control_L',True)
        self.x11.click(self.window,int(at),int((y+b)*.5))
        self.chord('a')
        for ch in str(value):
            key={'-':'minus','.':'period',' ':'space'}.get(ch,ch)
            self.event(key,True);self.event(key,False)
        self.event('Return',True);self.event('Return',False);self.x11.x.XSync(self.x11.display,False);time.sleep(.15)
    def drag(self,a,b):
        x=self.x11;x.move(self.window,*a);x.xt.XTestFakeButtonEvent(x.display,1,True,0);x.x.XSync(x.display,False)
        x.move(self.window,*b);x.xt.XTestFakeButtonEvent(x.display,1,False,0);x.x.XSync(x.display,False);time.sleep(.25)
