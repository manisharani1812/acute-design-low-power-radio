"""
Streaming IEEE 802.15.4 O-QPSK receiver (Python reference model).

Builds on oqpsk.py (chip table, half-sine, modulator). Adds what the ideal
loopback receiver lacked: frame detection, timing, carrier-frequency-offset
(CFO) estimation, phase/frequency tracking, SFD/PHR/FCS handling.

Streaming structure (maps to ACU kernels):
  detect_block   : sliding symbol-lag autocorrelation, explicit DetState
  acquire        : one-shot, runs on a short history buffer after a trigger
  track_symbol   : minimal-block kernel, one symbol (L + 2 samples) in,
                   one decision + loop updates out; loop state is explicit
Assumptions (not in the standard, stated for the study):
  * input at SPS = 4 samples/chip, unit signal amplitude (no AGC yet)
  * integer-sample timing resolution (+-0.5 sample = +-0.125 chip)
"""

from dataclasses import dataclass
import numpy as np
import oqpsk as ref

SPS = 4
CPS = ref.CHIPS_PER_SYMBOL            # 32 chips / symbol
L = CPS * SPS                         # samples per symbol = 128
FS = ref.CHIP_RATE * SPS              # 8 MS/s
TSYM = L / FS                         # 16 us
PULSE = ref.half_sine(SPS)
PM1 = ref.CHIP_TABLE_PM1              # (16, 32)
W_DET = 6 * L                         # detector window (6 symbols)
J_ACQ = 5                             # preamble symbols used in acquisition
M_HYP = (-2, -1, 0, 1, 2)             # CFO alias hypotheses, step FS/L = 62.5 kHz


# ----------------------------------------------------------------------
# Frame helpers (PPDU = SHR + PHR + PSDU)
# ----------------------------------------------------------------------

def crc16(data):
    """802.15.4 FCS: ITU-T CRC-16 (x^16+x^12+x^5+1), init 0, LSB first."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


def ppdu_bits(payload):
    """payload bytes (no FCS) -> PPDU bits (octets LSB first)."""
    mpdu = bytes(payload) + crc16(payload).to_bytes(2, "little")
    octets = [0, 0, 0, 0, 0xA7, len(mpdu)] + list(mpdu)
    return np.array([(o >> k) & 1 for o in octets for k in range(8)], dtype=np.int8)


# ----------------------------------------------------------------------
# Test channel: fractional timing, sample-clock offset, CFO, phase, AWGN
# ----------------------------------------------------------------------

def channel(bits, ebno_db, rng, cfo_hz=0.0, phase=0.0, frac=0.0,
            sco_ppm=0.0, pre=700, post=900):
    """Tx waveform -> rx stream at SPS. Noise power is set from unit signal
    power (NOT from the buffer mean, so idle padding does not bias Eb/N0)."""
    hi = 16
    d = hi // SPS
    xh = ref.modulate(bits, hi)
    xh = np.concatenate([np.zeros(pre * hi), xh, np.zeros(post * hi)])
    eps = sco_ppm * 1e-6
    n_out = int((len(xh) - 2 * d - 2) / (d * (1 + eps)))
    pos = frac * d + np.arange(n_out) * d * (1 + eps)
    m = np.arange(len(xh))
    x = np.interp(pos, m, xh.real) + 1j * np.interp(pos, m, xh.imag)
    n = np.arange(n_out)
    x = x * np.exp(1j * (2 * np.pi * cfo_hz * n / FS + phase))
    npow = (CPS / ref.BITS_PER_SYMBOL) * SPS / 10 ** (ebno_db / 10)
    x = x + np.sqrt(npow / 2) * (rng.standard_normal(n_out) + 1j * rng.standard_normal(n_out))
    return x


# ----------------------------------------------------------------------
# Kernel 1: sliding symbol-lag autocorrelation (frame detector)
# ----------------------------------------------------------------------

@dataclass
class DetState:
    xh: np.ndarray        # last L raw samples
    ch: np.ndarray        # last W products c[n] = x[n] x*[n-L]
    e1h: np.ndarray       # last W |x[n]|^2
    e2h: np.ndarray       # last W |x[n-L]|^2
    P: complex = 0j
    E1: float = 0.0
    E2: float = 0.0
    in_exc: bool = False
    maxM: float = 0.0
    idx_max: int = -1
    P_max: complex = 0j


def det_init():
    return DetState(np.zeros(L, complex), np.zeros(W_DET, complex),
                    np.zeros(W_DET), np.zeros(W_DET))


def detect_block(x, st):
    """Block in, per-sample metric M and complex P out; state carried."""
    B = len(x)
    xx = np.concatenate([st.xh, x])
    xl = xx[:B]                                   # x[n-L]
    c = x * np.conj(xl)
    e1, e2 = np.abs(x) ** 2, np.abs(xl) ** 2
    cc, ee1, ee2 = (np.concatenate([st.ch, c]), np.concatenate([st.e1h, e1]),
                    np.concatenate([st.e2h, e2]))
    P = st.P + np.cumsum(c - cc[:B])
    E1 = st.E1 + np.cumsum(e1 - ee1[:B])
    E2 = st.E2 + np.cumsum(e2 - ee2[:B])
    M = np.abs(P) ** 2 / np.maximum(E1 * E2, 1e-12)
    st.xh, st.ch, st.e1h, st.e2h = xx[-L:], cc[-W_DET:], ee1[-W_DET:], ee2[-W_DET:]
    st.P, st.E1, st.E2 = P[-1], E1[-1], E2[-1]
    return M, P


# ----------------------------------------------------------------------
# Chip extraction with derotation (shared by acquisition and tracking)
# ----------------------------------------------------------------------

def chips_from_window(xw, w0, g, phi_g, w, shift=0):
    """32 derotated, MF-normalised complex chips of the symbol starting at
    absolute sample g. xw holds raw samples starting at absolute index w0.
    Derotation phase: phi(n) = phi_g + w*(n-g)  (rad, rad/sample)."""
    k = np.arange(16)
    t = np.arange(2 * SPS)
    pk = g + shift + 2 * SPS - 1 + 2 * SPS * k          # I peak indices
    idx = np.concatenate([pk[:, None] - t[None, :], (pk + SPS)[:, None] - t[None, :]])
    x = xw[idx - w0] * np.exp(-1j * (phi_g + w * (idx - g)))
    v = (x * PULSE[None, :]).sum(axis=1) / SPS
    c = np.empty(CPS, complex)
    c[0::2] = v[:16]
    c[1::2] = v[16:] * (-1j)          # Q arm sits on the imaginary axis
    return c


# ----------------------------------------------------------------------
# Receiver
# ----------------------------------------------------------------------

@dataclass
class Frame:
    payload: bytes
    crc_ok: bool
    cfo_hz: float
    start: int


class Receiver:
    def __init__(self, thr=0.004, acq_thr=0.0075, accept=0.45, kp=0.12, ki=0.008, kt=0.15):
        self.thr, self.acq_thr, self.accept, self.kp, self.ki, self.kt = thr, acq_thr, accept, kp, ki, kt
        self.buf = np.zeros(0, complex)
        self.base = 0                  # absolute index of buf[0]
        self.n = 0                     # samples received
        self.det = det_init()
        self.det_n = 0                 # next sample to feed the detector
        self.mode = "SEARCH"
        self.frames = []
        self.n_false = 0
        self.warm_until = W_DET + L

    # -- public ---------------------------------------------------------
    def push(self, x):
        self.buf = np.concatenate([self.buf, np.asarray(x, complex)])
        self.n += len(x)
        progress = True
        while progress:
            progress = self._search() if self.mode == "SEARCH" else self._track()
        keep = self.n - 40 * L if self.mode == "SEARCH" else self.g - 3 * L
        if keep > self.base:
            self.buf = self.buf[keep - self.base:]
            self.base = keep

    # -- SEARCH: detector + excursion logic -------------------------------
    def _search(self):
        if self.det_n >= self.n:
            return False
        seg = self.buf[self.det_n - self.base:]
        st = self.det
        M, P = detect_block(seg, st)
        d0 = self.det_n
        above = np.flatnonzero(M > self.thr)
        i = 0 if st.in_exc else (int(above[0]) if above.size else len(M))
        while i < len(M):
            m = M[i]
            if d0 + i < self.warm_until:            # detector still filling its window
                i += 1
                continue
            if not st.in_exc:
                st.in_exc, st.maxM = True, 0.0
            if m > st.maxM:
                st.maxM, st.idx_max, st.P_max = m, d0 + i, P[i]
            elif m < 0.5 * st.maxM:
                n_pk, P_pk, big = st.idx_max, st.P_max, st.maxM >= self.acq_thr
                st.in_exc, st.maxM = False, 0.0
                if not big:                         # noise bump: drop silently
                    i += 1
                    continue
                if self._acquire(n_pk, P_pk):
                    self.det = det_init()
                    self.det_n = d0 + i + 1
                    self.warm_until = self.det_n + W_DET + L
                    return True
                self.n_false += 1
            i += 1
        self.det_n = self.n
        return False

    # -- ACQUIRE: CFO alias resolution + symbol timing --------------------
    def _acquire(self, n_pk, P_pk):
        f0 = np.angle(P_pk) * FS / (2 * np.pi * L)
        n0_lo = n_pk - 7 * L
        s0 = n0_lo - 2
        s1 = n0_lo + J_ACQ * L + 33 * SPS + 2
        if s0 < self.base or s1 > self.n:
            return False
        seg = self.buf[s0 - self.base: s1 - self.base]
        ai = np.arange(s0, s1)
        n0 = np.arange(L)[:, None, None]
        j = np.arange(J_ACQ)[None, :, None]
        k = np.arange(16)[None, None, :]
        pkI = n0_lo + n0 + j * L + 2 * SPS - 1 + 2 * SPS * k - s0     # (L,J,16)
        best = (-1.0, None)
        for m in M_HYP:
            fm = f0 + m * FS / L
            xd = seg * np.exp(-1j * 2 * np.pi * fm * ai / FS)
            y = np.convolve(xd, PULSE) / SPS
            c = np.empty((L, J_ACQ, CPS), complex)
            c[..., 0::2] = y[pkI]
            c[..., 1::2] = y[pkI + SPS] * (-1j)
            z0 = c @ PM1[0]                                           # (L,J)
            lam = np.abs(z0).sum(axis=1) / J_ACQ
            i = int(np.argmax(lam))
            if lam[i] > best[0]:
                best = (lam[i], (fm, i, z0[i].copy()))
        lam, (fm, i, z0) = best
        if lam < self.accept * CPS:                       # false-alarm rejection
            return False
        ph = np.unwrap(np.angle(z0))
        b, a = np.polyfit(np.arange(J_ACQ), ph, 1)      # rad/symbol, rad at sym 0 centre
        w_tot = 2 * np.pi * fm / FS + b / L
        g0 = n0_lo + i
        self.g = g0 + J_ACQ * L                         # first tracked symbol start
        # phase model phi(n) = phi_g + w (n-g); centre of sym0 has phase `a` after derotating fm
        centre0 = g0 + L / 2
        self.phi = 2 * np.pi * fm / FS * self.g + a + (b / L) * (self.g - centre0)
        self.w = w_tot
        self.t_acc = 0.0
        self.syms, self.stage, self.need = [], "SFD", 0
        self.start, self.cfo0 = g0, w_tot * FS / (2 * np.pi)
        self.mode = "TRACK"
        return True

    # -- TRACK: one symbol per step, explicit loop state -------------------
    def _track(self):
        if self.g + L + 3 * SPS > self.n:
            return False
        s = self.g - 2
        xw = self.buf[s - self.base:]
        w0 = s
        c0 = chips_from_window(xw, w0, self.g, self.phi, self.w, 0)
        z = PM1 @ c0
        sym = int(np.argmax(z.real))
        zs = z[sym]
        e = float(np.angle(zs))
        # early / late (timing)
        zp = (PM1[sym] @ chips_from_window(xw, w0, self.g, self.phi, self.w, +1)).real
        zm = (PM1[sym] @ chips_from_window(xw, w0, self.g, self.phi, self.w, -1)).real
        d = (zp - zm) / max(zs.real, 1e-9)
        self.t_acc += self.kt * float(np.clip(d, -1, 1))
        step = 0
        if self.t_acc > 0.5:
            step, self.t_acc = 1, self.t_acc - 1.0
        elif self.t_acc < -0.5:
            step, self.t_acc = -1, self.t_acc + 1.0
        # phase / frequency loop (rad, rad/symbol)
        om = self.w * L
        self.phi += om + self.kp * e
        self.w += self.ki * e / L
        self.g += L + step
        self._symbol(sym)
        return True

    def _symbol(self, sym):
        s = self.syms
        s.append(sym)
        if self.stage == "SFD":
            if len(s) >= 2 and s[-2] == 7 and s[-1] == 10:
                self.stage, self.syms = "PHR", []
            elif len(s) > 14:
                self._abort()
        elif self.stage == "PHR":
            if len(s) == 2:
                ln = (s[0] | (s[1] << 4)) & 0x7F
                if ln < 2:
                    return self._abort()
                self.stage, self.need, self.syms = "PSDU", 2 * ln, []
        elif self.stage == "PSDU" and len(s) == self.need:
            by = bytes(s[2 * i] | (s[2 * i + 1] << 4) for i in range(self.need // 2))
            ok = crc16(by[:-2]) == int.from_bytes(by[-2:], "little")
            self.frames.append(Frame(by[:-2], ok, self.w * FS / (2 * np.pi), self.start))
            self._abort(done=True)

    def _abort(self, done=False):
        self.mode = "SEARCH"
        self.det = det_init()
        self.det_n = max(self.g - L, self.base)
        self.warm_until = self.det_n + W_DET + L
