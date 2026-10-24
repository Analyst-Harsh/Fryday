import { useEffect, useRef, useState, type FormEvent } from "react";
import { percentile } from "./audio";
import { Session, TOKEN_KEY, type Snapshot, type Status, type TurnStats } from "./session";

// What the lamp says, in Fryday's language first.
const CAPTION: Record<Status, [string, string]> = {
  connecting: ["जुड़ रहा हूँ", "Connecting"],
  warming: ["तैयार हो रहा हूँ", "Getting ready: is `make llm` running?"],
  listening: ["सुन रहा हूँ", "Listening. Just talk."],
  hearing: ["सुन रहा हूँ", "Hearing you"],
  thinking: ["सोच रहा हूँ", "Thinking"],
  speaking: ["बोल रहा हूँ", "Speaking. Talk to interrupt."],
  offline: ["संपर्क टूट गया", "Can't reach Fryday. Retrying…"],
  replaced: ["दूसरे टैब में खुला है", "Fryday is open in another tab"],
  badToken: ["टोकन गलत है", "That token was refused"],
};

const CONNECTED: Status[] = ["listening", "hearing", "thinking", "speaking"];

function readToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function App() {
  const [token, setToken] = useState(readToken);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const session = useRef<Session | null>(null);

  useEffect(() => () => session.current?.stop(), []);

  async function begin(t: string) {
    setError(null);
    const s = new Session(t, setSnap);
    session.current = s;
    try {
      await s.start();
    } catch (e) {
      session.current = null;
      setSnap(null);
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Fryday needs the microphone. Allow it in the browser's site settings, then start again."
          : `Couldn't start audio: ${String(e)}`,
      );
    }
  }

  if (!snap || snap.status === "badToken") {
    return (
      <Welcome
        hasToken={Boolean(token) && snap?.status !== "badToken"}
        refused={snap?.status === "badToken"}
        error={error}
        onStart={(t) => {
          try {
            localStorage.setItem(TOKEN_KEY, t);
          } catch {
            // private mode: works for this visit only
          }
          setToken(t);
          void begin(t);
        }}
        savedToken={token}
      />
    );
  }
  return (
    <Conversation
      snap={snap}
      level={() => session.current?.level() ?? 0}
      onMute={() => session.current?.setMuted(!snap.muted)}
    />
  );
}

function Welcome(props: {
  hasToken: boolean;
  refused: boolean;
  error: string | null;
  savedToken: string;
  onStart: (token: string) => void;
}) {
  const [value, setValue] = useState(props.refused ? "" : props.savedToken);
  function submit(e: FormEvent) {
    e.preventDefault();
    if (value.trim()) props.onStart(value.trim());
  }
  return (
    <main className="welcome">
      <div className="lamp lamp--still" aria-hidden="true">
        <div className="lamp__glow" />
        <div className="lamp__core" />
      </div>
      <h1 className="wordmark wordmark--large">Fryday</h1>
      <p className="welcome__lede">Talk in Hinglish. Fryday answers out loud, and stops when you cut in.</p>
      <form className="welcome__form" onSubmit={submit}>
        {!props.hasToken && (
          <label className="field">
            <span className="field__label">Access token</span>
            <input
              className="field__input"
              type="password"
              autoComplete="current-password"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="FRYDAY_ACCESS_TOKEN from your .env"
              autoFocus
            />
          </label>
        )}
        {props.refused && <p className="welcome__error">That token was refused. Paste it again.</p>}
        {props.error && <p className="welcome__error">{props.error}</p>}
        <button className="button" type="submit" disabled={!value.trim()}>
          Start talking
        </button>
      </form>
      <p className="welcome__hint">Use headphones, or Fryday may hear itself.</p>
    </main>
  );
}

function Conversation(props: { snap: Snapshot; level: () => number; onMute: () => void }) {
  const { snap } = props;
  const lamp = useRef<HTMLDivElement>(null);
  const [drawer, setDrawer] = useState(false);
  const [devanagari, english] = CAPTION[snap.status];
  const last = snap.turns.at(-1);

  const level = useRef(props.level);
  level.current = props.level;

  // The lamp follows the voice: one rAF loop writes a CSS variable, React never re-renders.
  useEffect(() => {
    let frame = 0;
    let smooth = 0;
    const tick = () => {
      smooth += (level.current() - smooth) * 0.25;
      lamp.current?.style.setProperty("--level", smooth.toFixed(3));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <main className="room" data-state={snap.status}>
      <header className="bar">
        <span className="wordmark">Fryday</span>
        <button
          className="telemetry"
          type="button"
          onClick={() => setDrawer(!drawer)}
          aria-expanded={drawer}
          aria-controls="stats"
        >
          <span className={`dot ${CONNECTED.includes(snap.status) ? "dot--on" : ""}`} />
          <span className="telemetry__label">
            {CONNECTED.includes(snap.status) ? "Connected" : "Not connected"}
          </span>
          {snap.turns.length > 0 && <span className="telemetry__item">Turn {snap.turns.length}</span>}
          {last && <span className="telemetry__item">Reply in {seconds(last.heard_s)}</span>}
        </button>
      </header>

      <section className="stage">
        <div className="lamp" ref={lamp} aria-hidden="true">
          <div className="lamp__glow" />
          <div className="lamp__ring" />
          <div className="lamp__core" />
        </div>
        <p className="caption" aria-live="polite">
          <span className="caption__hi" lang="hi">
            {devanagari}
          </span>
          <span className="caption__en">{english}</span>
        </p>
      </section>

      <ol className="transcript" aria-label="Conversation">
        {snap.lines.map((line) => (
          <li key={line.id} className={`line line--${line.who}`} lang={line.who === "notice" ? "en" : "hi"}>
            {line.text}
            {line.interrupted && <span className="line__cut">…</span>}
          </li>
        ))}
      </ol>

      <footer className="controls">
        <button
          className={`mute ${snap.muted ? "mute--on" : ""}`}
          type="button"
          aria-pressed={snap.muted}
          onClick={props.onMute}
        >
          <MicIcon off={snap.muted} />
          <span>{snap.muted ? "Unmute" : "Mute"}</span>
        </button>
      </footer>

      {drawer && <StatsDrawer snap={snap} onClose={() => setDrawer(false)} />}
    </main>
  );
}

function StatsDrawer({ snap, onClose }: { snap: Snapshot; onClose: () => void }) {
  const last = snap.turns.at(-1);
  const totals = snap.turns.map((t) => t.heard_s);
  const browser = snap.turns.flatMap((t) => (t.browser_first_audio_s === null ? [] : [t.browser_first_audio_s]));
  const steps: [string, (t: TurnStats) => number | null][] = [
    ["Waiting for you to finish", (t) => t.vad_wait_s],
    ["Speech to text", (t) => t.asr_s],
    ["First word from the model", (t) => t.llm_first_token_s],
    ["First audio, server", (t) => t.first_audio_s],
    ["First audio, this browser", (t) => t.browser_first_audio_s],
  ];
  return (
    <aside className="drawer" id="stats" aria-label="Latency">
      <div className="drawer__head">
        <h2 className="drawer__title">How fast Fryday answered</h2>
        <button className="drawer__close" type="button" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      {last ? (
        <>
          <dl className="steps">
            {steps.map(([label, pick]) => (
              <div className="steps__row" key={label}>
                <dt>{label}</dt>
                <dd>{seconds(pick(last))}</dd>
              </div>
            ))}
            <div className="steps__row steps__row--total">
              <dt>You stop talking → Fryday speaks</dt>
              <dd>{seconds(last.heard_s)}</dd>
            </div>
          </dl>
          <p className="drawer__note">Last turn ({snap.turns.length} so far). Server times include the silence Fryday waits for.</p>
          <dl className="summary">
            <div>
              <dt>Typical</dt>
              <dd>{seconds(percentile(totals, 50))}</dd>
            </div>
            <div>
              <dt>Slowest 1 in 20</dt>
              <dd>{seconds(percentile(totals, 95))}</dd>
            </div>
            <div>
              <dt>Browser first audio, typical</dt>
              <dd>{seconds(percentile(browser, 50))}</dd>
            </div>
            <div>
              <dt>Interruptions</dt>
              <dd>{snap.interruptions}</dd>
            </div>
          </dl>
        </>
      ) : (
        <p className="drawer__note">Say something, and each step of the reply is timed here.</p>
      )}
      {Object.keys(snap.models).length > 0 && (
        <p className="drawer__models">
          {Object.entries(snap.models).map(([role, name]) => (
            <span key={role}>
              {role}: {name.split("/").at(-1)}
            </span>
          ))}
        </p>
      )}
    </aside>
  );
}

function MicIcon({ off }: { off: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
      {off && <path d="M4 4l16 16" />}
    </svg>
  );
}

function seconds(s: number | null | undefined): string {
  if (s === null || s === undefined) return "–";
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(2)} s`;
}
