import { describe, expect, it } from "vitest";
import { Downsampler, floatToPcm16, percentile, playedBytes, type Scheduled } from "./audio";

describe("Downsampler", () => {
  it("turns 48 kHz into a third as many samples, however the input is chunked", () => {
    const whole = new Downsampler(48_000, 16_000).push(new Float32Array(4800).fill(0.5));
    const chunked = new Downsampler(48_000, 16_000);
    let n = 0;
    for (const size of [128, 7, 1000, 3665]) n += chunked.push(new Float32Array(size).fill(0.5)).length;
    expect(whole.length).toBe(1600);
    expect(n).toBe(1600);
  });

  it("handles a fractional ratio (44.1 kHz) without drifting", () => {
    const d = new Downsampler(44_100, 16_000);
    let n = 0;
    for (let i = 0; i < 100; i++) n += d.push(new Float32Array(441)).length; // 1 s in 10 ms chunks
    expect(Math.abs(n - 16_000)).toBeLessThanOrEqual(1);
  });

  it("keeps a constant (DC) level and averages away a tone above 8 kHz", () => {
    const dc = new Downsampler(48_000, 16_000).push(new Float32Array(300).fill(0.25));
    expect(dc.every((s) => Math.abs(s - 0.25) < 1e-6)).toBe(true);
    // 16 kHz tone at 48 kHz: +1, -0.5, -0.5 repeating; each output averages one period.
    const tone = new Float32Array(300).map((_, i) => [1, -0.5, -0.5][i % 3]!);
    const out = new Downsampler(48_000, 16_000).push(tone);
    expect(Math.max(...out.map(Math.abs))).toBeLessThan(1e-6);
  });
});

describe("floatToPcm16", () => {
  it("clips and rounds like the server's to_pcm16", () => {
    expect([...floatToPcm16(new Float32Array([0, 0.5, 1, 1.2, -1, -1.5]))]).toEqual([
      0, 16384, 32767, 32767, -32767, -32767,
    ]);
  });
});

describe("playedBytes", () => {
  const chunks: Scheduled[] = [
    { start: 1, duration: 1, bytes: 2000 },
    { start: 2, duration: 1, bytes: 2000 },
  ];

  it("is exactly the bytes received once playback ends naturally", () => {
    expect(playedBytes(chunks, 3)).toBe(4000);
    expect(playedBytes(chunks, 99)).toBe(4000);
  });

  it("counts the played share of a chunk cut midway, in whole samples", () => {
    expect(playedBytes(chunks, 2.5)).toBe(3000);
    expect(playedBytes(chunks, 1.0001) % 2).toBe(0);
    expect(playedBytes(chunks, 0.5)).toBe(0);
  });
});

describe("percentile", () => {
  it("uses nearest rank", () => {
    const xs = [5, 1, 4, 2, 3];
    expect(percentile(xs, 50)).toBe(3);
    expect(percentile(xs, 95)).toBe(5);
    expect(percentile([], 50)).toBeNull();
  });
});
