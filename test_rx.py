"""Receiver tests: CFO sweep, block-size invariance, SCO, noise-only false alarms. Run: python3 test_rx.py"""
import numpy as np, oqpsk_rx as r
rng=np.random.default_rng(1)
pay=bytes(rng.integers(0,256,20).tolist())
print("CRC16 KERMIT('123456789') =",hex(r.crc16(b'123456789')),"(expect 0x2189)")
bits=r.ppdu_bits(pay)
def run(ebno,cfo,ph,frac,sco=0,blk=1000,seed=0):
    rg=np.random.default_rng(seed)
    x=r.channel(bits,ebno,rg,cfo_hz=cfo,phase=ph,frac=frac,sco_ppm=sco)
    rx=r.Receiver()
    for i in range(0,len(x),blk): rx.push(x[i:i+blk])
    return rx
for cfo in (0,30e3,-60e3,95e3,-150e3):
    rx=run(30,cfo,1.0,0.37)
    f=rx.frames
    print(f"noiseless-ish cfo {cfo/1e3:7.1f} kHz -> frames {len(f)}",
          [(x.payload==pay, x.crc_ok, round(x.cfo_hz/1e3,2)) for x in f], "false",rx.n_false)

import numpy as np, time, oqpsk_rx as r
rng=np.random.default_rng(5)
pay=bytes(rng.integers(0,256,20).tolist()); bits=r.ppdu_bits(pay)
# --- block-size invariance, noisy case
x=r.channel(bits,9,np.random.default_rng(11),cfo_hz=70e3,phase=2.0,frac=0.6)
res={}
for blk in (len(x),4096,1000,137,13,1):
    if blk==1: xx=x[:6000]; 
    else: xx=x
    rx=r.Receiver(); t=time.time()
    for i in range(0,len(xx),blk): rx.push(xx[i:i+blk])
    res[blk]=[(f.payload==pay,f.crc_ok,round(f.cfo_hz,1),f.start) for f in rx.frames]
    print("block",blk,res[blk],f"{time.time()-t:.2f}s")
# --- long frame + sample-clock offset
pay2=bytes(rng.integers(0,256,125).tolist()); bits2=r.ppdu_bits(pay2)
print("\n127-octet MPDU, 9 dB, CFO 80 kHz, SCO sweep")
for sco in (-80,-40,0,40,80):
    ok=0;N=8
    for s in range(N):
        xx=r.channel(bits2,9,np.random.default_rng(100+s),cfo_hz=80e3,phase=0.3*s,frac=0.1*s,sco_ppm=sco)
        rx=r.Receiver(); [rx.push(xx[i:i+2000]) for i in range(0,len(xx),2000)]
        ok+=sum(f.payload==pay2 and f.crc_ok for f in rx.frames)
    print(f"  SCO {sco:+4d} ppm : {ok}/{N} frames ok")

import numpy as np, time, oqpsk_rx as r, oqpsk as o
rng=np.random.default_rng(5)
pay=bytes(rng.integers(0,256,20).tolist()); bits=r.ppdu_bits(pay)
x=r.channel(bits,9,np.random.default_rng(11),cfo_hz=70e3,phase=2.0,frac=0.6)
rx=r.Receiver(); [rx.push(x[i:i+1]) for i in range(len(x))]
print("block=1 full stream:",[(f.payload==pay,f.crc_ok,round(f.cfo_hz,1),f.start) for f in rx.frames])
# control: is the timing loop doing anything? 127-octet frame, +80 ppm, kt=0 vs default
pay2=bytes(rng.integers(0,256,125).tolist()); bits2=r.ppdu_bits(pay2)
for kt in (0.0,0.15):
    ok=0
    for s in range(10):
        xx=r.channel(bits2,9,np.random.default_rng(300+s),cfo_hz=80e3,phase=0.3*s,frac=0.1*s,sco_ppm=80)
        rx=r.Receiver(kt=kt); [rx.push(xx[i:i+2000]) for i in range(0,len(xx),2000)]
        ok+=sum(f.payload==pay2 and f.crc_ok for f in rx.frames)
    print(f"timing gain kt={kt}: {ok}/10 long frames ok at +80 ppm")
# noise-only false alarms
rg=np.random.default_rng(9); N=1_500_000
xn=(rg.standard_normal(N)+1j*rg.standard_normal(N))*np.sqrt(0.5*32/10**0.7)   # noise at 7 dB Eb/N0 level
rx=r.Receiver(); [rx.push(xn[i:i+5000]) for i in range(0,N,5000)]
print(f"noise-only {N} samples ({N/r.FS*1e3:.0f} ms): frames {len(rx.frames)}, rejected acquisitions {rx.n_false}")
