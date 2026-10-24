// Reply playback: PCM16 chunks scheduled back to back on the AudioContext clock.
// Knows nothing about the socket: flush() only says how much was heard.

import { nextStart, pcm16ToFloat, playedBytes, type Scheduled } from "./audio";

export class Player {
  readonly analyser: AnalyserNode;
  sampleRate = 24_000; // replaced by ready.sample_rate
  private readonly out: GainNode;
  private queue: (Scheduled & { src: AudioBufferSourceNode; turn: number })[] = [];
  private playEnd = 0;

  constructor(
    private readonly ctx: AudioContext,
    private readonly onDrained: () => void, // the last scheduled chunk has finished
  ) {
    this.out = ctx.createGain();
    this.analyser = ctx.createAnalyser();
    this.out.connect(this.analyser);
    this.analyser.connect(ctx.destination);
  }

  /** Chunks of the current reply, played or not (cleared only by flush or a new turn). */
  get hasQueued(): boolean {
    return this.queue.length > 0;
  }

  /** Schedules a chunk of `turn`; returns seconds until it starts sounding. */
  play(pcm: ArrayBuffer, turn: number): number {
    const ctx = this.ctx;
    const samples = pcm16ToFloat(pcm);
    const buffer = ctx.createBuffer(1, samples.length, this.sampleRate); // the browser resamples
    buffer.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.out);
    const start = nextStart(this.playEnd, ctx.currentTime);
    src.start(start);
    this.playEnd = start + buffer.duration;
    if (this.queue.length && this.queue[0]!.turn !== turn) this.queue = []; // a new reply
    this.queue.push({ src, turn, start, duration: buffer.duration, bytes: pcm.byteLength });
    src.onended = () => {
      if (ctx.currentTime >= this.playEnd - 0.01) this.onDrained();
    };
    return start - ctx.currentTime;
  }

  /**
   * Stops all reply audio. `cut`: audio was still playing (a real interruption).
   * `bytes`: PCM16 bytes heard so far, or null if nothing was queued.
   */
  flush(): { cut: boolean; bytes: number | null } {
    const now = this.ctx.currentTime;
    const cut = this.queue.some((c) => now < c.start + c.duration);
    const bytes = this.queue.length ? playedBytes(this.queue, now) : null;
    for (const c of this.queue) {
      try {
        c.src.stop();
      } catch {
        // never started
      }
    }
    this.queue = [];
    this.playEnd = 0;
    return { cut, bytes };
  }
}
