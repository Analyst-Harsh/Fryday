// Pure audio maths for the browser client, kept out of the worklet and React so it's testable.

/**
 * Streams audio from the context's rate (e.g. 48 kHz, or 44.1 kHz on some Macs) down to
 * 16 kHz. Each output sample is the mean of the input samples it covers: averaging first is
 * a crude low-pass, so frequencies above 8 kHz don't fold back as noise (aliasing), and
 * position carries across calls, so chunk boundaries leave no clicks or drift.
 * ponytail: a box filter, not a proper windowed-sinc; fine for speech into Whisper.
 */
export class Downsampler {
  private readonly ratio: number;
  private pending = new Float32Array(0); // input not yet covered by an output sample
  private pos = 0; // where the next output sample starts, in `pending` (fractional)

  constructor(fromRate: number, toRate: number) {
    if (toRate > fromRate) throw new Error(`can't upsample ${fromRate} → ${toRate}`);
    this.ratio = fromRate / toRate;
  }

  push(input: Float32Array): Float32Array {
    const buf = new Float32Array(this.pending.length + input.length);
    buf.set(this.pending);
    buf.set(input, this.pending.length);
    const out = new Float32Array(Math.max(0, Math.floor((buf.length - this.pos) / this.ratio)));
    let n = 0;
    while (n < out.length) {
      const start = Math.floor(this.pos);
      const end = Math.min(buf.length, Math.floor(this.pos + this.ratio));
      let sum = 0;
      for (let i = start; i < end; i++) sum += buf[i]!;
      out[n++] = sum / Math.max(1, end - start);
      this.pos += this.ratio;
    }
    const used = Math.floor(this.pos);
    this.pending = buf.slice(used);
    this.pos -= used;
    return out;
  }
}

/** Float samples in [-1, 1] → PCM16, clipped and rounded like the server's tts.to_pcm16. */
export function floatToPcm16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]!));
    out[i] = Math.round(s * 32767);
  }
  return out;
}

export function pcm16ToFloat(pcm: ArrayBuffer): Float32Array<ArrayBuffer> {
  const ints = new Int16Array(pcm);
  const out = new Float32Array(ints.length);
  for (let i = 0; i < ints.length; i++) out[i] = ints[i]! / 32768;
  return out;
}

/** A chunk of reply audio scheduled on the AudioContext clock. */
export interface Scheduled {
  start: number; // seconds, context time
  duration: number;
  bytes: number; // PCM16 bytes as received
}

/** When the next chunk should start: right after the last one, or a little ahead of now
 * (a 50 ms jitter buffer) if playback has run dry. */
export const nextStart = (prevEnd: number, now: number, lead = 0.05): number =>
  Math.max(prevEnd, now + lead);

/**
 * PCM16 bytes of these chunks heard by `now`. Once every chunk has finished, this is exactly
 * the bytes received, so the server keeps the whole reply; a cut mid-chunk counts the
 * share played, in whole samples.
 */
export function playedBytes(chunks: Scheduled[], now: number): number {
  let total = 0;
  for (const c of chunks) {
    if (now >= c.start + c.duration) total += c.bytes;
    else if (now > c.start) total += 2 * Math.floor(((c.bytes / 2) * (now - c.start)) / c.duration);
  }
  return total;
}

/** Root-mean-square level of an analyser's waveform, 0 (silence) to ~1 (full scale). */
export function rms(samples: Float32Array): number {
  let sum = 0;
  for (const s of samples) sum += s * s;
  return Math.sqrt(sum / Math.max(1, samples.length));
}

/** The p-th percentile (0–100) by nearest rank, or null for no data. */
export function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1]!;
}
