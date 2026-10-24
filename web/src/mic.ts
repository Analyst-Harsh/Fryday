// The microphone graph: mic → analyser (for the lamp) and mic → worklet (20 ms batches).

export interface Mic {
  stream: MediaStream;
  analyser: AnalyserNode;
  node: AudioWorkletNode; // held so the worklet lives as long as the session
}

/** Opens the mic on `ctx` and calls `onSamples` with raw batches at the context's rate. */
export async function openMic(ctx: AudioContext, onSamples: (samples: Float32Array) => void): Promise<Mic> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  await ctx.audioWorklet.addModule("/mic-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "mic-capture", { numberOfOutputs: 0 });
  const analyser = ctx.createAnalyser();
  source.connect(analyser);
  source.connect(node);
  node.port.onmessage = (e: MessageEvent<Float32Array>) => onSamples(e.data);
  return { stream, analyser, node };
}
