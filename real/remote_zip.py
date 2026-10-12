import urllib.request, struct, sys, zlib, io
U="https://data.cnra.ca.gov/dataset/5e2d49e1-9ed0-425e-9f3e-2cda4a213c26/resource/6e855ae7-f365-4d40-984f-3c016b54f69d/download/verticaldisplacementpointdata.zip"
import subprocess
def rng(a,b):
    p=subprocess.run(["curl","-sSL","--max-time","600","-r",f"{a}-{b}","-D","/dev/stderr",U],capture_output=True)
    cr=[l for l in p.stderr.decode(errors="replace").splitlines() if l.lower().startswith("content-range")]
    return p.stdout,(cr[-1].split(":",1)[1].strip() if cr else None)
def size():
    _,cr=rng(0,0); return int(cr.split("/")[1])
def entries():
    N=size(); tail,_=rng(N-200000,N-1)
    i=tail.rfind(b"PK\x05\x06")
    if i<0: raise SystemExit("no EOCD")
    cd_size,cd_off=struct.unpack("<II",tail[i+12:i+20])
    n=struct.unpack("<H",tail[i+10:i+12])[0]
    # zip64?
    j=tail.rfind(b"PK\x06\x06")
    if j>=0:
        n,cd_size,cd_off=struct.unpack("<QQQ",tail[j+32:j+56])
    cd,_=rng(cd_off,cd_off+cd_size-1)
    out=[];p=0
    while p<len(cd) and cd[p:p+4]==b"PK\x01\x02":
        (meth,)=struct.unpack("<H",cd[p+10:p+12])
        csz,usz=struct.unpack("<II",cd[p+20:p+28])
        nl,el,cl=struct.unpack("<HHH",cd[p+28:p+34])
        (loff,)=struct.unpack("<I",cd[p+42:p+46])
        name=cd[p+46:p+46+nl].decode("utf-8","replace")
        ex=cd[p+46+nl:p+46+nl+el]
        q=0
        while q<len(ex):
            hid,hl=struct.unpack("<HH",ex[q:q+4]); d=ex[q+4:q+4+hl]
            if hid==1:
                k=0
                if usz==0xFFFFFFFF: usz=struct.unpack("<Q",d[k:k+8])[0];k+=8
                if csz==0xFFFFFFFF: csz=struct.unpack("<Q",d[k:k+8])[0];k+=8
                if loff==0xFFFFFFFF: loff=struct.unpack("<Q",d[k:k+8])[0];k+=8
            q+=4+hl
        out.append((name,meth,csz,usz,loff)); p+=46+nl+el+cl
    return out
def extract(e,dest):
    name,meth,csz,usz,loff=e
    h,_=rng(loff,loff+29)
    nl,el=struct.unpack("<HH",h[26:30])
    start=loff+30+nl+el
    d=zlib.decompressobj(-15) if meth==8 else None
    with open(dest,"wb") as f:
        pos=start;end=start+csz
        while pos<end:
            b,_=rng(pos,min(end,pos+64*1024*1024)-1); pos+=len(b)
            f.write(d.decompress(b) if d else b)
        if d: f.write(d.flush())
if __name__=="__main__":
    es=entries()
    print(len(es))
    for e in es[:400]: print(e[0],e[1],e[2],e[3])
