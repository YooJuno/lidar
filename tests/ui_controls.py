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
    def right_click(self,name):
        x,y,r,b=self.rect(name);self.x11.click(self.window,int((x+r)*.5),int((y+b)*.5),3);time.sleep(.1)
    def double_click(self,name):
        x,y,r,b=self.rect(name)
        self.x11.move(self.window,int((x+r)*.5),int((y+b)*.5))
        for _ in range(2):
            self.x11.xt.XTestFakeButtonEvent(self.x11.display,1,True,0)
            self.x11.xt.XTestFakeButtonEvent(self.x11.display,1,False,0)
            self.x11.x.XSync(self.x11.display,False);time.sleep(.08)
        time.sleep(.2)
    def roi_action(self,action,index=None):
        after=len(self.log.read_text(errors='replace'))
        self.right_click('roi_selected_row'if index is None else f'roi_row_{index}')
        name=f'roi_rename_{index}'if action=='rename'and index is not None else f'roi_{action}'
        x,y,r,b=self.rect(name,after=after);self.x11.click(self.window,int((x+r)*.5),int((y+b)*.5));time.sleep(.1)
    def roi_toggle_outlines(self):
        self.roi_action('details');self.click('roi_outlines');self.click('roi_done')
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
