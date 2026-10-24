// Runs on the audio thread: batches 128-sample render quanta into ~20 ms and posts them.
// Downsampling happens on the main thread (src/audio.ts), where it can be unit-tested.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    // 20 ms at the context rate. Kept apart from the array: once posted, its buffer is
    // transferred and the old array reads as length 0 (a 0-length batch loops forever).
    this.size = Math.round(sampleRate / 50);
    this.batch = new Float32Array(this.size);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    let i = 0;
    while (i < channel.length) {
      const n = Math.min(channel.length - i, this.size - this.filled);
      this.batch.set(channel.subarray(i, i + n), this.filled);
      this.filled += n;
      i += n;
      if (this.filled === this.size) {
        this.port.postMessage(this.batch, [this.batch.buffer]); // transfer, don't copy
        this.batch = new Float32Array(this.size);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
