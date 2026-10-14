import numpy as np, oqpsk as o
# Python translation of gr-advoqpsk decode_impl::fSigGenRI (sps=2, their pulse [sin(i*pi/4)], i=0..3)
SIN=np.sin(np.arange(4)/4*np.pi)
def ref_wave(byte):
    lo,hi=byte%16,byte//16
    R=np.r_[o.CHIP_TABLE[lo][0::2],o.CHIP_TABLE[hi][0::2]]   # even chips -> R(I)
    Q=np.r_[o.CHIP_TABLE[lo][1::2],o.CHIP_TABLE[hi][1::2]]   # odd chips  -> I(Q)
    r=np.zeros(130); q=np.zeros(130)
    for i in range(32):
        r[i*4:i*4+4]=SIN*(1 if R[i] else -1)
        q[i*4+2:i*4+6]=SIN*(1 if Q[i] else -1)
    r[128:]=0; q[:2]=0
    return (r+1j*q)[:128]
rng=np.random.default_rng(3)
data=rng.integers(0,256,100)
x=np.concatenate([ref_wave(int(b)) for b in data])
# (a) chip-stream equivalence: reference byte->chips vs ours (bits LSB-first per octet)
bits=np.array([(b>>k)&1 for b in data for k in range(8)],dtype=np.int8)
ours=o.modulate(bits,sps=2)
print("sample counts ref/ours:",x.size,ours.size)
# (b) decode reference-generated waveform with our receiver
rx=o.demodulate_ideal(np.r_[x,np.zeros(8)],2,n_symbols=2*len(data))
print("our Rx on reference-style Tx waveform, bit errors:",int(np.sum(rx!=bits)),"of",bits.size)
# (c) decode our waveform with the reference's own decision rule (I at 4i+3, Q at 4i+5 after MF with their pulse)
mf=lambda s: np.convolve(s,SIN)
c_i=mf(ours.real)[3::4][:3200]; c_q=mf(ours.imag)[5::4][:3200]
ref_chips=np.empty(6400); ref_chips[0::2]=c_i; ref_chips[1::2]=c_q
sym,_=o.despread(np.sign(ref_chips)); 
print("reference-style sampling on our Tx, bit errors:",int(np.sum(o.symbols_to_bits(sym)!=bits)))
