// One conversation with Fryday: the WebSocket, the turn state machine and the Snapshot.
// Plain TypeScript (no React), so the UI only renders snapshots of it.
// Audio lives in mic.ts (capture) and player.ts (playback); both share this AudioContext.

import { Downsampler, floatToPcm16, rms } from "./audio";
import { openMic, type Mic } from "./mic";
import { Player } from "./player";
import { CLOSE, hello, MIC_RATE, parse, played, type ServerMessage, type Stats } from "./protocol";

export type Status =
  | "connecting"
  | "warming" // server up, models or the model server not ready yet
  | "listening"
  | "hearing" // the VAD heard you start
  | "thinking"
  | "speaking"
  | "offline"
  | "replaced" // another tab took over
  | "badToken";

export interface Line {
  id: string;
  who: "you" | "fryday" | "notice";
  text: string;
  turn: number;
  interrupted?: boolean;
}

export interface TurnStats extends Stats {
  turn: number;
  browser_first_audio_s: number | null; // thinking → first sound, measured here
}

export interface Snapshot {
  status: Status;
  lines: Line[];
  turns: TurnStats[]; // most recent last, up to 20
  interruptions: number;
  models: Record<string, string>;
  muted: boolean;
}

const MAX_LINES = 12;
const MAX_TURNS = 20;

export class Session {
  private snap: Snapshot = {
    status: "connecting",
    lines: [],
    turns: [],
    interruptions: 0,
    models: {},
    muted: false,
  };
  private ws: WebSocket | null = null;
  private ctx: AudioContext | null = null;
  private mic: Mic | null = null;
  private player: Player | null = null;
  private down: Downsampler | null = null;
  private ready = false;
  private replyTurn: number | null = null; // the turn of the last reply event
  private thinkingAt: number | null = null; // for the browser-side first-audio time
  private firstAudio = new Map<number, number>();
  private retryMs = 1000;
  private stopped = false;
  private wave = new Float32Array(1024);

  constructor(
    private token: string,
    private readonly onChange: (snap: Snapshot) => void,
  ) {}

  /** Must run from a click: browsers only start audio after a user gesture. */
  async start(): Promise<void> {
    this.ctx = new AudioContext(); // the device's own rate; we downsample ourselves
    await this.ctx.resume();
    this.mic = await openMic(this.ctx, (samples) => this.onMic(samples));
    this.down = new Downsampler(this.ctx.sampleRate, MIC_RATE);
    this.player = new Player(this.ctx, () => {
      if (this.snap.status === "speaking") this.update({ status: "listening" });
    });
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    this.ws?.close();
    this.mic?.stream.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
  }

  setMuted(muted: boolean): void {
    this.mic?.stream.getAudioTracks().forEach((t) => (t.enabled = !muted));
    this.update({ muted });
  }

  /** 0..1, for the lamp: Fryday's voice while speaking, otherwise yours. */
  level(): number {
    const analyser = this.snap.status === "speaking" ? this.player?.analyser : this.mic?.analyser;
    if (!analyser) return 0;
    analyser.getFloatTimeDomainData(this.wave);
    return Math.min(1, rms(this.wave) * 4);
  }

  private connect(): void {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    this.update({ status: "connecting" });
    ws.onopen = () => ws.send(hello(this.token)); // the token only ever travels here
    ws.onmessage = (e: MessageEvent<string | ArrayBuffer>) => {
      if (typeof e.data === "string") {
        const msg = parse(e.data);
        if (msg) this.onMessage(msg);
      } else {
        this.play(e.data);
      }
    };
    ws.onclose = (e) => this.onClose(e.code);
  }

  private onClose(code: number): void {
    this.ready = false;
    this.player?.flush(); // the socket is gone: nobody to report to
    if (code === CLOSE.badToken) {
      localStorage.removeItem(TOKEN_KEY);
      this.update({ status: "badToken" });
      return;
    }
    if (code === CLOSE.replaced) {
      this.update({ status: "replaced" });
      return;
    }
    if (this.stopped) return;
    this.update({ status: code === CLOSE.notReady ? "warming" : "offline" });
    window.setTimeout(() => this.connect(), this.retryMs);
    this.retryMs = Math.min(5000, this.retryMs * 2);
  }

  private onMic(samples: Float32Array): void {
    if (!this.ready || this.snap.muted || this.ws?.readyState !== WebSocket.OPEN || !this.down) {
      return;
    }
    const pcm = floatToPcm16(this.down.push(samples));
    if (pcm.length) this.ws.send(pcm.buffer);
  }

  private onMessage(msg: ServerMessage): void {
    switch (msg.t) {
      case "ready":
        this.ready = true;
        this.retryMs = 1000;
        if (this.player) this.player.sampleRate = msg.sample_rate;
        this.update({ status: "listening", models: msg.models });
        break;
      case "speech": // you started talking: stop Fryday now (barge-in)
        if (this.bargeIn() && this.replyTurn !== null) {
          // Usually the server had already finished the turn, so no "interrupted" event
          // comes: mark the cut reply here.
          this.markInterrupted(this.replyTurn);
          this.update({ interruptions: this.snap.interruptions + 1 });
        }
        this.update({ status: "hearing" });
        break;
      case "thinking":
        this.thinkingAt = performance.now();
        this.update({ status: "thinking" });
        break;
      case "transcript":
        this.addLine({ who: "you", text: msg.text, turn: msg.turn });
        break;
      case "reply":
        this.replyTurn = msg.turn;
        this.appendReply(msg.turn, msg.text);
        break;
      case "done": {
        const stats: TurnStats = {
          ...msg.stats,
          turn: msg.turn,
          browser_first_audio_s: this.firstAudio.get(msg.turn) ?? null,
        };
        this.update({ turns: [...this.snap.turns, stats].slice(-MAX_TURNS) });
        if (!this.player?.hasQueued) this.update({ status: "listening" });
        break;
      }
      case "interrupted":
        this.markInterrupted(msg.turn);
        break;
      case "notice":
        this.addLine({ who: "notice", text: msg.text, turn: msg.turn ?? -1 });
        if (this.snap.status === "hearing" || this.snap.status === "thinking") {
          this.update({ status: "listening" });
        }
        break;
    }
  }

  private play(pcm: ArrayBuffer): void {
    if (!this.player || this.replyTurn === null) return;
    const turn = this.replyTurn;
    const delay = this.player.play(pcm, turn);
    if (!this.firstAudio.has(turn) && this.thinkingAt !== null) {
      this.firstAudio.set(turn, (performance.now() - this.thinkingAt) / 1000 + delay);
    }
    this.update({ status: "speaking" });
  }

  /**
   * Stop all reply audio and tell the server how much of the last reply was heard, so its
   * history keeps only that. Sent right away, before the audio that ends your utterance,
   * so it always arrives before the next turn starts. True if audio was still playing.
   */
  private bargeIn(): boolean {
    const { cut, bytes } = this.player?.flush() ?? { cut: false, bytes: null };
    if (this.replyTurn !== null && bytes !== null) this.ws?.send(played(this.replyTurn, bytes));
    return cut;
  }

  private markInterrupted(turn: number): void {
    const lines = this.snap.lines.map((l) =>
      l.who === "fryday" && l.turn === turn ? { ...l, interrupted: true } : l,
    );
    this.update({ lines });
  }

  private addLine(line: Omit<Line, "id">): void {
    const lines = [...this.snap.lines, { ...line, id: crypto.randomUUID() }].slice(-MAX_LINES);
    this.update({ lines });
  }

  private appendReply(turn: number, text: string): void {
    const last = this.snap.lines.at(-1);
    if (last?.who === "fryday" && last.turn === turn) {
      const lines = [...this.snap.lines.slice(0, -1), { ...last, text: `${last.text} ${text}` }];
      this.update({ lines });
    } else {
      this.addLine({ who: "fryday", text, turn });
    }
  }

  private update(patch: Partial<Snapshot>): void {
    this.snap = { ...this.snap, ...patch };
    this.onChange(this.snap);
  }
}

export const TOKEN_KEY = "fryday.token";
