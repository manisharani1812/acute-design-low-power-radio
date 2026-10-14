"""
IEEE 802.15.4 2.4 GHz O-QPSK / DSSS -- Python reference model.

Project 27 -- Modulation and demodulation kernels for low-power radio.
Milestone: deterministic transmitter + ideal-channel loopback demodulator.

Processing stages are pure functions: inputs are explicit and outputs
are returned directly, making the stages suitable for later porting
to streaming kernels on the ACU.

PHY parameters:
    4 bits per symbol
    16 symbols
    32 chips per symbol
    2.0 Mchip/s
    62.5 ksymbol/s
    250 kb/s
    Half-sine pulse spanning 2 chip periods
    Q arm delayed by one chip period relative to I
"""

import numpy as np


# ----------------------------------------------------------------------
# PHY constants
# ----------------------------------------------------------------------

CHIP_RATE = 2_000_000
CHIPS_PER_SYMBOL = 32
BITS_PER_SYMBOL = 4
SYMBOL_COUNT = 16


# IEEE 802.15.4 2.4 GHz PHY chip sequences.
_CHIP_STRINGS = [
    "11011001110000110101001000101110",
    "11101101100111000011010100100010",
    "00101110110110011100001101010010",
    "00100010111011011001110000110101",
    "01010010001011101101100111000011",
    "00110101001000101110110110011100",
    "11000011010100100010111011011001",
    "10011100001101010010001011101101",
    "10001100100101100000011101111011",
    "10111000110010010110000001110111",
    "01111011100011001001011000000111",
    "01110111101110001100100101100000",
    "00000111011110111000110010010110",
    "01100000011101111011100011001001",
    "10010110000001110111101110001100",
    "11001001011000000111011110111000",
]

CHIP_TABLE = np.array(
    [[int(c) for c in s] for s in _CHIP_STRINGS],
    dtype=np.int8,
)

CHIP_TABLE_PM1 = (2 * CHIP_TABLE - 1).astype(np.float64)


def _validate_sps(sps):
    """Require a positive integer number of samples per chip."""
    if (
        isinstance(sps, (bool, np.bool_))
        or not isinstance(sps, (int, np.integer))
        or sps < 1
    ):
        raise ValueError("sps must be a positive integer")


def verify_chip_table():
    """Check the cyclic-shift and odd-chip inversion properties."""
    ok = True

    for k in range(8):
        ok &= np.array_equal(
            np.roll(CHIP_TABLE[0], 4 * k),
            CHIP_TABLE[k],
        )
        ok &= np.array_equal(
            np.roll(CHIP_TABLE[8], 4 * k),
            CHIP_TABLE[8 + k],
        )

    conjugate = CHIP_TABLE[0].copy()
    conjugate[1::2] ^= 1

    ok &= np.array_equal(conjugate, CHIP_TABLE[8])

    return bool(ok)


# ----------------------------------------------------------------------
# Transmitter
# ----------------------------------------------------------------------

def bits_to_symbols(bits):
    """Pack groups of four bits into symbols, least-significant bit first."""
    bits = np.asarray(bits)

    if bits.ndim != 1:
        raise ValueError("bits must be a one-dimensional array")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1")

    if bits.size % BITS_PER_SYMBOL != 0:
        raise ValueError("bit count must be a multiple of 4")

    bits = bits.astype(np.int8, copy=False)

    nibbles = bits.reshape(-1, BITS_PER_SYMBOL)
    weights = (1 << np.arange(BITS_PER_SYMBOL)).astype(np.int8)

    return (nibbles @ weights).astype(np.int8)


def symbols_to_bits(symbols):
    """Convert symbols 0..15 into bits, least-significant bit first."""
    symbols = np.asarray(symbols)

    if symbols.ndim != 1:
        raise ValueError("symbols must be a one-dimensional array")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must contain finite values")

    if not np.all((symbols >= 0) & (symbols < SYMBOL_COUNT)):
        raise ValueError("symbols must be integers from 0 to 15")

    if not np.all(symbols == np.floor(symbols)):
        raise ValueError("symbols must contain integers")

    symbols = symbols.astype(np.int8, copy=False)
    shifts = np.arange(BITS_PER_SYMBOL)

    return (
        ((symbols[:, None] >> shifts) & 1)
        .reshape(-1)
        .astype(np.int8)
    )


def spread(symbols):
    """Map each symbol to its 32-chip sequence, represented as +/-1."""
    symbols = np.asarray(symbols)

    if symbols.ndim != 1:
        raise ValueError("symbols must be a one-dimensional array")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must contain finite values")

    if not np.all((symbols >= 0) & (symbols < SYMBOL_COUNT)):
        raise ValueError("symbols must be integers from 0 to 15")

    if not np.all(symbols == np.floor(symbols)):
        raise ValueError("symbols must contain integers")

    symbols = symbols.astype(np.int8, copy=False)

    return CHIP_TABLE_PM1[symbols].reshape(-1)


def half_sine(sps):
    """Generate a symmetric half-sine pulse spanning two chip periods."""
    _validate_sps(sps)

    n = np.arange(2 * sps)

    return np.sin(
        np.pi * (n + 0.5) / (2.0 * sps)
    )


def chips_to_baseband(chips, sps):
    """
    Convert +/-1 chips into complex O-QPSK baseband samples.

    Even-index chips use the I arm.
    Odd-index chips use the Q arm, delayed by one chip period.
    """
    _validate_sps(sps)

    chips = np.asarray(chips, dtype=np.float64)

    if chips.ndim != 1:
        raise ValueError("chips must be a one-dimensional array")

    if not np.all(np.isfinite(chips)):
        raise ValueError("chips must contain finite values")

    if not np.all((chips == 1.0) | (chips == -1.0)):
        raise ValueError("chips must contain only -1 and +1")

    if chips.size % 2 != 0:
        raise ValueError("chip count must be even (I and Q arms)")

    if chips.size == 0:
        return np.empty(0, dtype=np.complex128)

    a_i = chips[0::2]
    a_q = chips[1::2]

    n_arm = a_i.size
    pulse = half_sine(sps)
    step = 2 * sps

    # Allocate enough space for the complete last pulse.
    length = (n_arm - 1) * step + sps + len(pulse)

    imp_i = np.zeros(length, dtype=np.float64)
    imp_q = np.zeros(length, dtype=np.float64)

    imp_i[np.arange(n_arm) * step] = a_i
    imp_q[sps + np.arange(n_arm) * step] = a_q

    i_wave = np.convolve(imp_i, pulse, mode="full")[:length]
    q_wave = np.convolve(imp_q, pulse, mode="full")[:length]

    return i_wave + 1j * q_wave


def modulate(bits, sps=8):
    """Convert input bits into complex baseband samples."""
    _validate_sps(sps)

    symbols = bits_to_symbols(bits)
    chips = spread(symbols)

    return chips_to_baseband(chips, sps)


# ----------------------------------------------------------------------
# Channel
# ----------------------------------------------------------------------

def awgn(x, ebno_db, sps, rng=None):
    """Add complex AWGN using the requested Eb/N0."""
    _validate_sps(sps)

    x = np.asarray(x, dtype=np.complex128)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain finite samples")

    if not np.isfinite(ebno_db):
        raise ValueError("ebno_db must be finite")

    if x.size == 0:
        return x.copy()

    rng = np.random.default_rng() if rng is None else rng

    samples_per_bit = (
        CHIPS_PER_SYMBOL / BITS_PER_SYMBOL
    ) * sps

    signal_power = np.mean(np.abs(x) ** 2)
    ebno_linear = 10.0 ** (ebno_db / 10.0)

    if signal_power == 0:
        return x.copy()

    noise_power = signal_power * samples_per_bit / ebno_linear

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(x.size)
        + 1j * rng.standard_normal(x.size)
    )

    return x + noise


# ----------------------------------------------------------------------
# Receiver -- ideal channel, known timing
# ----------------------------------------------------------------------

def sample_chips(x, sps, n_chips):
    """Matched-filter the I/Q arms and sample each chip pulse."""
    _validate_sps(sps)

    x = np.asarray(x, dtype=np.complex128)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain finite samples")

    if (
        isinstance(n_chips, (bool, np.bool_))
        or not isinstance(n_chips, (int, np.integer))
        or n_chips < 0
        or n_chips % 2 != 0
    ):
        raise ValueError("n_chips must be a non-negative even integer")

    if n_chips == 0:
        return np.empty(0, dtype=np.float64)

    pulse = half_sine(sps)
    n_arm = n_chips // 2

    step = 2 * sps
    base = 2 * sps - 1

    idx_i = base + step * np.arange(n_arm)
    idx_q = base + sps + step * np.arange(n_arm)

    required_length = max(idx_i[-1], idx_q[-1]) + 1

    if x.size < required_length:
        raise ValueError(
            f"signal is too short: need at least {required_length} samples"
        )

    mf_i = np.convolve(np.real(x), pulse, mode="full")
    mf_q = np.convolve(np.imag(x), pulse, mode="full")

    chips = np.empty(n_chips, dtype=np.float64)

    chips[0::2] = mf_i[idx_i]
    chips[1::2] = mf_q[idx_q]

    # The energy of the sampled half-sine pulse is sps.
    return chips / sps


def despread(chips):
    """Correlate each 32-chip block against all 16 chip sequences."""
    chips = np.asarray(chips, dtype=np.float64)

    if chips.ndim != 1:
        raise ValueError("chips must be a one-dimensional array")

    if not np.all(np.isfinite(chips)):
        raise ValueError("chips must contain finite values")

    if chips.size % CHIPS_PER_SYMBOL != 0:
        raise ValueError("chip count must be a multiple of 32")

    if chips.size == 0:
        return (
            np.empty(0, dtype=np.int8),
            np.empty((0, SYMBOL_COUNT), dtype=np.float64),
        )

    blocks = chips.reshape(-1, CHIPS_PER_SYMBOL)
    metrics = blocks @ CHIP_TABLE_PM1.T

    symbols = np.argmax(metrics, axis=1).astype(np.int8)

    return symbols, metrics


def demodulate_ideal(x, sps, n_symbols):
    """Demodulate with known timing and no carrier-frequency offset."""
    _validate_sps(sps)

    if (
        isinstance(n_symbols, (bool, np.bool_))
        or not isinstance(n_symbols, (int, np.integer))
        or n_symbols < 0
    ):
        raise ValueError("n_symbols must be a non-negative integer")

    x = np.asarray(x, dtype=np.complex128)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")

    if n_symbols == 0:
        return np.empty(0, dtype=np.int8)

    n_chips = n_symbols * CHIPS_PER_SYMBOL

    chips = sample_chips(x, sps, n_chips)
    symbols, _ = despread(chips)

    return symbols_to_bits(symbols)


# ----------------------------------------------------------------------
# Verification helpers
# ----------------------------------------------------------------------

def check_envelope(x, sps, guard=None):
    """Return minimum and maximum envelope magnitude away from transients."""
    _validate_sps(sps)

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")

    g = 2 * sps if guard is None else guard

    if (
        isinstance(g, (bool, np.bool_))
        or not isinstance(g, (int, np.integer))
        or g < 0
    ):
        raise ValueError("guard must be a non-negative integer")

    if x.size == 0 or 2 * g >= x.size:
        raise ValueError("signal is too short for the requested guard")

    mag = np.abs(x[g:-g] if g else x)

    return float(mag.min()), float(mag.max())


def check_phase_steps(x, sps, guard=None):
    """Return per-sample phase increments in units of pi/(2*sps)."""
    _validate_sps(sps)

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")

    g = 2 * sps if guard is None else guard

    if (
        isinstance(g, (bool, np.bool_))
        or not isinstance(g, (int, np.integer))
        or g < 0
    ):
        raise ValueError("guard must be a non-negative integer")

    if x.size < 3 or 2 * g >= x.size:
        raise ValueError("signal is too short for the requested guard")

    guarded = x[g:-g] if g else x

    phase = np.unwrap(np.angle(guarded))

    return np.diff(phase) / (np.pi / (2 * sps))


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(27)
    SPS = 8

    print(
        "chip table structure :",
        "PASS" if verify_chip_table() else "FAIL",
    )

    bits = rng.integers(0, 2, 4 * 200).astype(np.int8)

    x = modulate(bits, sps=SPS)

    print(f"bits                 : {bits.size}")
    print(f"samples              : {x.size} ({SPS} samples/chip)")

    lo, hi = check_envelope(x, SPS)

    print(f"envelope |x|         : min {lo:.9f} max {hi:.9f}")

    steps = check_phase_steps(x, SPS)
    uniq = np.unique(np.round(steps, 6))

    print(f"phase increments     : {uniq}")

    rx = demodulate_ideal(
        x,
        SPS,
        n_symbols=bits.size // BITS_PER_SYMBOL,
    )

    errors = int(np.count_nonzero(bits != rx))

    print(
        f"loopback             : "
        f"{'PASS' if errors == 0 else 'FAIL'} "
        f"({errors} bit errors)"
    )

    # Explicit edge-case tests.
    print("\nInput edge cases")

    for name, test_bits in [
        ("empty input", []),
        ("3 bits", [1, 0, 1]),
        ("5 bits", [1, 0, 1, 0, 1]),
        ("invalid bit", [0, 1, 2, 0]),
    ]:
        try:
            result = modulate(test_bits, sps=SPS)
            print(f"  {name:15s}: PASS ({result.size} samples)")
        except (ValueError, TypeError) as exc:
            print(f"  {name:15s}: REJECTED ({exc})")

    # BER simulation.
    print("\nBER vs Eb/N0 (ideal timing, no CFO)")

    n_bits = 4 * 20000

    for ebno_db in range(0, 11, 2):
        b = rng.integers(0, 2, n_bits).astype(np.int8)

        tx = modulate(b, sps=SPS)
        y = awgn(tx, ebno_db, SPS, rng=rng)

        r = demodulate_ideal(
            y,
            SPS,
            n_symbols=n_bits // BITS_PER_SYMBOL,
        )

        ber = np.mean(b != r)

        print(
            f"  Eb/N0 {ebno_db:2d} dB   "
            f"BER {ber:.2e} "
            f"{'(no errors)' if ber == 0 else ''}"
        )
