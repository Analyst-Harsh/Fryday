// WebSocket protocol v1, the browser side (server: app/src/fryday/connection.py).
// Text frames are JSON {"v": 1, "t": <type>, ...}; binary frames are PCM16 LE mono audio.

export const PROTOCOL = 1;
export const MIC_RATE = 16_000; // what the server's VAD and Whisper expect

export const CLOSE = {
  restart: 1012, // the server is restarting: reconnect
  notReady: 1013, // models loading or no model server: reconnect
  badMessage: 4400,
  badToken: 4401,
  noHello: 4408,
  replaced: 4409, // a newer tab took over: don't reconnect
} as const;

export interface Stats {
  vad_wait_s: number;
  asr_s: number;
  llm_first_token_s: number | null;
  first_audio_s: number | null;
  speech_in_s: number;
  heard_s: number; // you stop talking → Fryday speaks, on the server
}

export type ServerMessage =
  | { t: "ready"; sample_rate: number; turn: number; models: Record<string, string> }
  | { t: "speech"; turn: number }
  | { t: "thinking"; turn: number }
  | { t: "transcript"; turn: number; text: string }
  | { t: "reply"; turn: number; text: string }
  | { t: "done"; turn: number; stats: Stats }
  | { t: "interrupted"; turn: number }
  | { t: "notice"; code: string; text: string; turn?: number };

const TYPES = new Set<string>([
  "ready",
  "speech",
  "thinking",
  "transcript",
  "reply",
  "done",
  "interrupted",
  "notice",
]);

/** A server message, or null if it isn't JSON of our protocol version with a known type. */
export function parse(text: string): ServerMessage | null {
  let msg: unknown;
  try {
    msg = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof msg !== "object" || msg === null) return null;
  const { v, t } = msg as { v?: unknown; t?: unknown };
  if (v !== PROTOCOL || typeof t !== "string" || !TYPES.has(t)) return null;
  return msg as ServerMessage;
}

export const hello = (token: string): string => JSON.stringify({ v: PROTOCOL, t: "hello", token });

export const played = (turn: number, bytes: number): string =>
  JSON.stringify({ v: PROTOCOL, t: "played", turn, bytes });
