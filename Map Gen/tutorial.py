#!/usr/bin/env python3
"""For Loop Tutorial — 540x960 vertical, 30fps, 30s"""
import os, math, subprocess, wave, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H   = 540, 960
FPS    = 30
DUR    = 30
NFRAM  = FPS * DUR   # 900
OUT    = "/sessions/wizardly-keen-gauss/mnt/outputs"
V_FILE = f"{OUT}/fl_video.mp4"
A_FILE = f"{OUT}/fl_audio.wav"
F_FILE = f"{OUT}/forloop_tutorial.mp4"

WHITE  = (255,255,255); BLUE   = (88,166,255);  YELLOW = (230,192,123)
GREEN  = (152,195,121); GREEN2 = (100,220,100); RED    = (224,108,117)
PURPLE = (198,120,221); MUTED  = (139,148,158); CARD   = (22,27,34)
BORDER = (48,54,61)

FD = "/usr/share/fonts/truetype"
_FP = {'title':f"{FD}/google-fonts/Poppins-Bold.ttf",
       'bold': f"{FD}/dejavu/DejaVuSans-Bold.ttf",
       'reg':  f"{FD}/dejavu/DejaVuSans.ttf",
       'mono': f"{FD}/dejavu/DejaVuSansMono.ttf",
       'monob':f"{FD}/dejavu/DejaVuSansMono-Bold.ttf"}
_FC={}
def F(k,s):
    if (k,s) not in _FC: _FC[(k,s)]=ImageFont.truetype(_FP[k],s)
    return _FC[(k,s)]

_ya=np.arange(H,dtype=np.float32)
_ba=np.zeros((H,W,3),dtype=np.uint8)
_ba[:,:,0]=np.clip(13+_ya[:,None]*6/H,0,255).astype(np.uint8)
_ba[:,:,1]=np.clip(17+_ya[:,None]*4/H,0,255).astype(np.uint8)
_ba[:,:,2]=np.clip(23+_ya[:,None]*10/H,0,255).astype(np.uint8)
_BGI=Image.fromarray(_ba,'RGB')
def nf(): return _BGI.copy()

def eio(t):
    t=max(0.0,min(1.0,t)); return t*t*(3-2*t)
def fade(f,s,d): return eio((f-s)/max(d,1))
def tw(d,t,fn):
    bb=d.textbbox((0,0),t,font=fn); return bb[2]-bb[0]
def th(d,t,fn):
    bb=d.textbbox((0,0),t,font=fn); return bb[3]-bb[1]
def cxt(d,txt,y,fn,col,al):
    if al<=0: return
    w_=tw(d,txt,fn); x=(W-w_)//2; r,g,b=col
    d.text((x,y),txt,font=fn,fill=(r,g,b,int(al*255)))
def comp(base,dfn):
    ov=Image.new('RGBA',(W,H),(0,0,0,0)); dv=ImageDraw.Draw(ov); dfn(dv)
    res=base.convert('RGBA'); res.alpha_composite(ov); return res.convert('RGB')
def afade(img,f):
    zones=[(138,152,1.0,0.0),(152,166,0.0,1.0),(438,452,1.0,0.0),(452,466,0.0,1.0),
           (738,752,1.0,0.0),(752,766,0.0,1.0),(882,899,1.0,0.0)]
    for s,e,a0,a1 in zones:
        if s<=f<e:
            t=(f-s)/(e-s); al=a0+(a1-a0)*eio(t)
            if al<1.0:
                arr=np.array(img,dtype=np.float32)
                img=Image.fromarray((arr*al).astype(np.uint8))
            break
    return img

# ─── SCENE 1: TITLE (frames 0-151) ───────────────────────────
def s1(f):
    img=nf()
    def draw(d):
        for gx in range(40,W,50):
            for gy in range(40,H,50):
                d.ellipse([gx-1,gy-1,gx+1,gy+1],fill=BORDER+(45,))
        a=fade(f,0,22); dy=int((1-eio(min(1,f/22)))*28)
        ft=F('title',65); w_=tw(d,"For Loops",ft); x=(W-w_)//2
        d.text((x+2,361-dy+2),"For Loops",font=ft,fill=BLUE+(int(a*50),))
        d.text((x,361-dy),"For Loops",font=ft,fill=WHITE+(int(a*255),))
        a2=fade(f,18,22); dy2=int((1-eio(min(1,max(0,(f-18)/22))))*18)
        fs=F('title',49); w2=tw(d,"in Python",fs)
        d.text(((W-w2)//2,443-dy2),"in Python",font=fs,fill=BLUE+(int(a2*255),))
        a3=fade(f,38,18); lw=int(a3*280); cx_=W//2
        if lw>0: d.line([(cx_-lw//2,506),(cx_+lw//2,506)],fill=BLUE+(int(a3*200),),width=2)
        a4=fade(f,55,22); ft2=F('reg',26)
        cxt(d,"Repeat code with ease",521,ft2,MUTED,a4)
        a5=fade(f,78,20)
        if a5>0:
            ox,oy=W//2,612
            for i in range(6):
                ang=i*(math.pi/3)+f*0.025
                ex=ox+int(28*math.cos(ang)); ey=oy+int(28*math.sin(ang)); sz=7-i//2
                d.ellipse([ex-sz,ey-sz,ex+sz,ey+sz],fill=GREEN+(int(a5*200),))
    return comp(img,draw)

# ─── SCENE 2: CONCEPT (frames 150-451) ───────────────────────
ITEMS=["apple","banana","cherry"]; ICOLS=[RED,YELLOW,GREEN]
BXX,BWW,BHH,BGAP=(W-340)//2,340,59,79; BY0=262

def s2(f):
    lf=f-150; img=nf()
    PS=80; ptr_y=BY0+BHH//2
    if lf>=PS:
        idx=min(2,(lf-PS)//55)
        base_y=BY0+idx*BGAP+BHH//2
        ns=PS+(idx+1)*55
        if idx<2 and lf>=ns-12:
            t2=eio(min(1,(lf-(ns-12))/12))
            base_y=int((BY0+idx*BGAP+BHH//2)*(1-t2)+(BY0+(idx+1)*BGAP+BHH//2)*t2)
        ptr_y=base_y
    def draw(d):
        a=fade(lf,0,18); fh=F('bold',34); cxt(d,"What is a for loop?",97,fh,YELLOW,a)
        a2=fade(lf,15,18); fd=F('reg',24)
        for ii,ln in enumerate(["Go through each item in a","list, one by one."]):
            cxt(d,ln,162+ii*32,fd,WHITE,a2*0.85)
        for i,(item,col) in enumerate(zip(ITEMS,ICOLS)):
            ba=fade(lf,28+i*18,15)
            if ba<=0: continue
            by=BY0+i*BGAP; hs=PS+i*55; he=hs+50
            is_hl=(lf>=PS) and (hs<=lf<he)
            r_,g_,b_=col
            bf=(r_//4,g_//4,b_//4,int(ba*235)) if is_hl else CARD+(int(ba*235),)
            bo=col+(int(ba*(255 if is_hl else 180)),)
            d.rounded_rectangle([BXX,by,BXX+BWW,by+BHH],radius=10,fill=bf,outline=bo,width=2)
            fi=F('bold',32); tw_=tw(d,item,fi); tx=(W-tw_)//2
            ty=by+(BHH-th(d,item,fi))//2
            d.text((tx,ty),item,font=fi,fill=(col if is_hl else WHITE)+(int(ba*255),))
        if lf>=PS-5:
            pa=fade(lf,PS-5,12)
            if pa>0:
                tx2=BXX-11; ax0=tx2-65; ay=ptr_y
                d.line([(ax0,ay),(tx2-11,ay)],fill=PURPLE+(int(pa*220),),width=4)
                d.polygon([(tx2-11,ay-10),(tx2,ay),(tx2-11,ay+10)],fill=PURPLE+(int(pa*220),))
                fl=F('bold',19); tw_=tw(d,"loop",fl)
                arrow_cx=(ax0+tx2)//2; d.text((arrow_cx-tw_//2,ay-28),"loop",font=fl,fill=PURPLE+(int(pa*200),))
    return comp(img,draw)

# ─── SCENE 3: CODE (frames 450-751) ──────────────────────────
CL=[
    (18,[("fruits",WHITE,False),(" = [",MUTED,False)]),
    (25,[('    "apple",',GREEN,False)]),
    (32,[('    "banana",',GREEN,False)]),
    (39,[('    "cherry"',GREEN,False)]),
    (46,[("]",MUTED,False)]),
    (53,[]),
    (58,[("for ",PURPLE,True),("fruit ",WHITE,False),("in ",PURPLE,True),("fruits:",WHITE,False)]),
    (68,[("    ",WHITE,False),("print",BLUE,False),("(fruit)",WHITE,False)]),
]
def s3(f):
    lf=f-450; img=nf()
    def draw(d):
        a=fade(lf,0,18); fh=F('bold',37); cxt(d,"The Code",75,fh,YELLOW,a)
        ca=fade(lf,12,20)
        if ca>0:
            x1,y1,x2,y2=31,142,W-31,487
            d.rounded_rectangle([x1,y1,x2,y2],radius=11,fill=CARD+(int(ca*248),),outline=BORDER+(int(ca*255),),width=2)
            for ii,dc in enumerate([RED,YELLOW,GREEN]):
                ex=x1+17+ii*21; ey=y1+15
                d.ellipse([ex-6,ey-6,ex+6,ey+6],fill=dc+(int(ca*200),))
            fm=F('mono',22); fmb=F('monob',22)
            cx_=x1+26; cy_=y1+40; lnh=40
            for li,(ls,segs) in enumerate(CL):
                la=fade(lf,ls,10)
                if la<=0 or not segs: continue
                xp=cx_
                for st,sc,sb in segs:
                    sf=fmb if sb else fm; r_,g_,b_=sc
                    d.text((xp,cy_+li*lnh),st,font=sf,fill=(r_,g_,b_,int(la*255)))
                    bb=d.textbbox((0,0),st,font=sf); xp+=bb[2]-bb[0]
        oa=fade(lf,92,18)
        if oa>0:
            fo=F('bold',26); cxt(d,"Output:",500,fo,GREEN,oa)
            fov=F('mono',24)
            for ii,val in enumerate(["apple","banana","cherry"]):
                va=fade(lf,108+ii*20,14)
                if va>0: cxt(d,f">>> {val}",537+ii*41,fov,GREEN2,va)
    return comp(img,draw)

# ─── SCENE 4: SUMMARY (frames 750-899) ───────────────────────
PTS=[("for loops","iterate over sequences"),("each item","is visited once"),("clean syntax","easy to read")]
def s4(f):
    lf=f-750; img=nf()
    def draw(d):
        a=fade(lf,0,20); fh=F('bold',40); cxt(d,"You learned:",117,fh,WHITE,a)
        for i,(kw,desc) in enumerate(PTS):
            pa=fade(lf,25+i*22,18)
            if pa<=0: continue
            py=210+i*81; sx=int((1-eio(min(1,max(0,(lf-25-i*22)/18))))*32)
            d.ellipse([47-sx,py+8,63-sx,py+24],fill=BLUE+(int(pa*255),))
            fkw=F('bold',27); fdsc=F('reg',27); kws=f"{kw}:"
            d.text((75-sx,py),kws,font=fkw,fill=BLUE+(int(pa*255),))
            kw_w=tw(d,kws,fkw)
            d.text((75-sx+kw_w+7,py),desc,font=fdsc,fill=WHITE+(int(pa*255),))
        ha=fade(lf,80,22)
        if ha>0:
            pulse=0.75+0.25*math.sin(lf*0.18)
            ft=F('title',49); txt="Happy Coding!"; tw_=tw(d,txt,ft); tx=(W-tw_)//2
            for gr in [28,18,8]:
                d.text((tx,522),txt,font=ft,fill=YELLOW+(int(ha*pulse*gr*4),))
            d.text((tx,522),txt,font=ft,fill=YELLOW+(int(ha*255),))
        ca=fade(lf,105,18)
        if ca>0:
            fc=F('reg',20); cxt(d,"Follow for more Python tips!",590,fc,MUTED,ca*0.85)
    return comp(img,draw)

def rf(f):
    if   f<152: img=s1(f)
    elif f<452: img=s2(f)
    elif f<752: img=s3(f)
    else:       img=s4(f)
    return afade(img,f)

# ─── AUDIO ────────────────────────────────────────────────────
def gen_audio(dur=DUR,sr=44100):
    print("Generating background music...")
    n=dur*sr; out=np.zeros(n,dtype=np.float32)
    chords=[[261.63,329.63,392.00],[220.00,261.63,329.63],
            [174.61,220.00,261.63],[196.00,246.94,293.66]]
    clen=3.0
    for ci in range(int(dur/clen)+2):
        ch=chords[ci%len(chords)]; s=int(ci*clen*sr); e=min(int((ci+1)*clen*sr),n)
        if s>=n: break
        m=e-s; lt=np.linspace(0,clen,m,dtype=np.float32)
        env=np.ones(m,dtype=np.float32)
        atk=min(int(0.35*sr),m//4); rel=min(int(0.35*sr),m//4)
        env[:atk]=np.linspace(0,1,atk); env[-rel:]=np.linspace(1,0,rel)
        for freq in ch:
            out[s:e]+=0.10*env*np.sin(2*np.pi*freq*lt)
            out[s:e]+=0.04*env*np.sin(2*np.pi*freq*2*lt)
            out[s:e]+=0.015*env*np.sin(2*np.pi*freq*3*lt)
    fi=int(1.2*sr); fo=int(1.8*sr)
    out[:fi]*=np.linspace(0,1,fi); out[-fo:]*=np.linspace(1,0,fo)
    mx=np.max(np.abs(out))
    if mx>0: out=out/mx*0.55
    with wave.open(A_FILE,'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((out*32767).astype(np.int16).tobytes())
    print(f"  Saved {A_FILE}")

# ─── VIDEO ────────────────────────────────────────────────────
def gen_video():
    print("Rendering frames → ffmpeg...")
    cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
         '-s',f'{W}x{H}','-pix_fmt','rgb24','-r',str(FPS),'-i','pipe:0',
         '-vcodec','libx264','-pix_fmt','yuv420p','-preset','fast','-crf','20',V_FILE]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for f in range(NFRAM):
        if f%90==0: print(f"  Frame {f}/{NFRAM}  ({f//FPS}s)",flush=True)
        img=rf(f)
        proc.stdin.write(np.array(img,dtype=np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"  Saved {V_FILE}")

def merge():
    print("Merging video + audio...")
    subprocess.run(['ffmpeg','-y','-i',V_FILE,'-i',A_FILE,
                    '-c:v','copy','-c:a','aac','-b:a','192k','-shortest',F_FILE],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print(f"  Final: {F_FILE}")

if __name__=='__main__':
    gen_audio()
    gen_video()
    merge()
    sz=os.path.getsize(F_FILE)
    print(f"\nDone! {F_FILE}  ({sz//1024} KB)")
