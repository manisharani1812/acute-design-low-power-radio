import numpy as np, time, oqpsk_rx as r, oqpsk as o
N=200
rng=np.random.default_rng(2026)
print("PER, 20-octet payload (23-octet MPDU+PHR), CFO ~U(+-100 kHz), random phase/timing")
print(" Eb/N0   rx PER  (missed / crc-fail)   ideal-genie PER   n")
t0=time.time()
for e in (4,5,6,7,8,9):
    miss=crcf=ok=0; ideal_err=0
    for _ in range(N):
        pay=bytes(rng.integers(0,256,20).tolist()); bits=r.ppdu_bits(pay)
        x=r.channel(bits,e,rng,cfo_hz=rng.uniform(-100e3,100e3),phase=rng.uniform(0,2*np.pi),frac=rng.uniform(0,1))
        rx=r.Receiver(); [rx.push(x[i:i+2000]) for i in range(0,len(x),2000)]
        good=[f for f in rx.frames if f.crc_ok and f.payload==pay]
        if good: ok+=1
        elif rx.frames: crcf+=1
        else: miss+=1
        # genie: ideal timing/CFO, ideal receiver on PHR+PSDU bits (after 5 SHR octets = 40 bits)
        y=o.awgn(o.modulate(bits,sps=4),e,4,rng=rng)
        rb=o.demodulate_ideal(y,4,n_symbols=bits.size//4)
        ideal_err+= np.any(rb[40:]!=bits[40:])
    per=1-ok/N
    print(f"  {e} dB   {per:6.3f}   ({miss:3d} / {crcf:3d})            {ideal_err/N:6.3f}        {N}   [{time.time()-t0:.0f}s]",flush=True)
