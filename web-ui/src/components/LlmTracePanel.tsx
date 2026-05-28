import { useEffect, useMemo, useState } from 'react';
import type {
  LlmTraceContentPart,
  LlmTraceMessage,
  LlmTraceRecord,
  LlmTraceResponse,
  SessionState,
} from '../types/messages';

interface LlmTracePanelProps {
  sessionId: string | null;
  state: SessionState;
}

function formatTime(ts?: string): string {
  if (!ts) return 'unknown time';
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return ts;
  return date.toLocaleTimeString();
}

function countWords(text?: string | null): number {
  if (!text) return 0;
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function callLabel(record: LlmTraceRecord, index: number): string {
  const call = record.llm_call ?? index + 1;
  const phase = record.phase || 'unknown';
  const turn = record.turn ?? 0;
  const model = record.model || 'model';
  return `Call #${call} · ${phase} · turn ${turn} · ${model}`;
}

function partText(part: LlmTraceContentPart): string {
  if ('type' in part && part.type === 'text') {
    return typeof part.text === 'string' ? part.text : '';
  }
  if ('type' in part && part.type === 'image_url') {
    const ref = typeof part.image_ref === 'string' ? part.image_ref : 'saved image';
    const chars = typeof part.elided_chars === 'number' ? `, ${part.elided_chars} chars elided` : '';
    return `[image: ${ref}${chars}]`;
  }
  return JSON.stringify(part, null, 2);
}

function messageContent(content: LlmTraceMessage['content']): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map(partText).filter(Boolean).join('\n\n');
  }
  if (content == null) return '';
  return JSON.stringify(content, null, 2);
}

function roleStyle(role: string): string {
  switch (role) {
    case 'system':
      return 'border-l-blue-400/60 bg-blue-950/20';
    case 'user':
      return 'border-l-accent/70 bg-accent/5';
    case 'assistant':
      return 'border-l-nv-green/60 bg-nv-green/5';
    default:
      return 'border-l-text-tertiary/60 bg-surface-raised/60';
  }
}

function TextBubble({
  label,
  value,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  tone?: 'neutral' | 'reasoning' | 'content';
}) {
  if (!value.trim()) return null;
  const toneClass =
    tone === 'reasoning'
      ? 'border-l-purple-400/60 bg-purple-950/15'
      : tone === 'content'
      ? 'border-l-nv-green/60 bg-nv-green/5'
      : 'border-l-text-tertiary/60 bg-surface-raised/60';

  return (
    <section className={`rounded-md border border-surface-border border-l-2 ${toneClass} overflow-hidden`}>
      <div className="flex items-center justify-between gap-3 px-3 py-2 border-b border-surface-border bg-surface-sunken/40">
        <span className="text-[11px] font-display font-semibold uppercase tracking-wide text-text-secondary">
          {label}
        </span>
        <span className="text-[11px] font-mono text-text-muted">
          {value.length.toLocaleString()} chars
        </span>
      </div>
      <div className="px-3 py-3 text-sm leading-relaxed text-text-primary whitespace-pre-wrap break-words">
        {value}
      </div>
    </section>
  );
}

function InputMessage({ message, index }: { message: LlmTraceMessage; index: number }) {
  const content = messageContent(message.content);
  return (
    <section className={`rounded-md border border-surface-border border-l-2 ${roleStyle(message.role)} overflow-hidden`}>
      <div className="flex items-center justify-between gap-3 px-3 py-2 border-b border-surface-border bg-surface-sunken/40">
        <div className="flex items-center gap-2 min-w-0">
          <span className="w-5 h-5 rounded bg-surface-overlay border border-surface-border flex items-center justify-center text-[10px] font-mono text-text-tertiary">
            {index + 1}
          </span>
          <span className="text-[11px] font-display font-semibold uppercase tracking-wide text-text-secondary">
            {message.role || 'message'}
          </span>
        </div>
        <span className="text-[11px] font-mono text-text-muted">
          {content.length.toLocaleString()} chars
        </span>
      </div>
      <div className="px-3 py-3 text-sm leading-relaxed text-text-primary whitespace-pre-wrap break-words">
        {content || <span className="text-text-tertiary italic">Empty message</span>}
      </div>
    </section>
  );
}

export function LlmTracePanel({ sessionId, state }: LlmTracePanelProps) {
  const [records, setRecords] = useState<LlmTraceRecord[]>([]);
  const [tracePath, setTracePath] = useState<string | null>(null);
  const [exists, setExists] = useState(false);
  const [selectedCall, setSelectedCall] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedRecord = useMemo(() => {
    if (records.length === 0) return null;
    if (selectedCall == null) return records[records.length - 1];
    return records.find((record, index) => (record.llm_call ?? index + 1) === selectedCall) ?? records[records.length - 1];
  }, [records, selectedCall]);

  const refreshTrace = async (quiet = false) => {
    if (!sessionId) {
      setRecords([]);
      setTracePath(null);
      setExists(false);
      return;
    }
    if (!quiet) setLoading(true);
    try {
      const response = await fetch(`/api/session/${sessionId}/llm-trace`);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || 'Could not load LLM trace');
      }
      const data = (await response.json()) as LlmTraceResponse;
      setRecords(data.records);
      setTracePath(data.path);
      setExists(data.exists);
      setError(null);
      setSelectedCall((current) => {
        if (data.records.length === 0) return null;
        if (current != null && data.records.some((record, index) => (record.llm_call ?? index + 1) === current)) {
          return current;
        }
        const latest = data.records[data.records.length - 1];
        return latest.llm_call ?? data.records.length;
      });
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Unknown trace error');
    } finally {
      if (!quiet) setLoading(false);
    }
  };

  useEffect(() => {
    refreshTrace();
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || !['loading_config', 'running', 'awaiting_user_input'].includes(state)) return;
    const interval = window.setInterval(() => {
      refreshTrace(true);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [sessionId, state]);

  const outputContent = selectedRecord?.output?.content || '';
  const outputReasoning = selectedRecord?.output?.reasoning || '';

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-surface">
      <div className="flex-shrink-0 px-5 py-4 border-b border-surface-border bg-surface-raised">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-accent/10 border border-accent/20 flex items-center justify-center">
                <svg className="w-3.5 h-3.5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7h16M4 12h16M4 17h10" />
                </svg>
              </div>
              <h2 className="text-sm font-display font-bold text-text-primary tracking-wide">LLM Trace</h2>
            </div>
            <p className="mt-1 text-xs text-text-tertiary truncate">
              {tracePath || 'Start a trial to browse llm_trace.jsonl'}
            </p>
          </div>
          <button
            onClick={() => refreshTrace()}
            disabled={!sessionId || loading}
            className="flex items-center justify-center w-8 h-8 rounded-md border border-surface-border text-text-tertiary hover:text-text-primary hover:bg-surface-overlay disabled:opacity-40 transition-colors"
            title="Refresh trace"
            aria-label="Refresh trace"
          >
            <svg className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v6h6M20 20v-6h-6M5 19A9 9 0 0119 5M19 5h-5M5 19h5" />
            </svg>
          </button>
        </div>

        <div className="mt-4 flex items-center gap-3">
          <select
            value={selectedRecord ? selectedRecord.llm_call ?? records.indexOf(selectedRecord) + 1 : ''}
            onChange={(event) => setSelectedCall(Number(event.target.value))}
            disabled={records.length === 0}
            className="min-w-0 flex-1 appearance-none px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-xs font-display text-text-primary focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40"
          >
            {records.length === 0 ? (
              <option value="">No LLM calls yet</option>
            ) : (
              records.map((record, index) => {
                const value = record.llm_call ?? index + 1;
                return (
                  <option key={`${value}-${record._line ?? index}`} value={value}>
                    {callLabel(record, index)}
                  </option>
                );
              })
            )}
          </select>
          <span className="flex-shrink-0 text-xs font-mono text-text-tertiary">
            {records.length} rows
          </span>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-5">
        {!sessionId ? (
          <div className="h-full flex items-center justify-center text-sm text-text-tertiary">
            Start or reconnect to a trial first.
          </div>
        ) : error ? (
          <div className="rounded-md border border-red-800/30 bg-red-950/30 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        ) : !exists || records.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center gap-3">
            <div className="w-10 h-10 rounded-md bg-surface-raised border border-surface-border flex items-center justify-center">
              <svg className="w-5 h-5 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6M7 4h10l3 3v13H4V7l3-3z" />
              </svg>
            </div>
            <div>
              <div className="text-sm font-display text-text-secondary">Waiting for trace rows</div>
              <div className="text-xs text-text-tertiary mt-1">The panel refreshes while the trial is running.</div>
            </div>
          </div>
        ) : selectedRecord?.parse_error ? (
          <TextBubble label={`JSON parse error on line ${selectedRecord._line ?? '?'}`} value={selectedRecord.raw || selectedRecord.parse_error} />
        ) : selectedRecord ? (
          <div className="space-y-5">
            <section className="rounded-md border border-surface-border bg-surface-raised overflow-hidden">
              <div className="px-3 py-2 border-b border-surface-border flex items-center justify-between gap-2">
                <span className="text-[11px] font-display font-semibold uppercase tracking-wide text-text-secondary">
                  Metadata
                </span>
                <span className="text-[11px] font-mono text-text-muted">{formatTime(selectedRecord.ts)}</span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-px bg-surface-border text-xs">
                {[
                  ['Call', `#${selectedRecord.llm_call ?? '?'}`],
                  ['Phase', selectedRecord.phase || '?'],
                  ['Turn', String(selectedRecord.turn ?? '?')],
                  ['Model', selectedRecord.model || '?'],
                  ['Duration', selectedRecord.duration_s == null ? '?' : `${selectedRecord.duration_s}s`],
                  ['Decision', selectedRecord.decision || '?'],
                  ['Code', String(selectedRecord.code_blocks?.length ?? 0)],
                  ['Line', String(selectedRecord._line ?? '?')],
                ].map(([label, value]) => (
                  <div key={label} className="bg-surface-raised px-3 py-2 min-w-0">
                    <div className="text-text-muted font-display uppercase tracking-wide text-[10px]">{label}</div>
                    <div className="text-text-primary truncate mt-0.5">{value}</div>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-display font-bold uppercase tracking-wide text-accent">Input</h3>
                <span className="text-xs text-text-tertiary">
                  {selectedRecord.input_messages?.length ?? 0} messages
                </span>
              </div>
              <div className="space-y-3">
                {(selectedRecord.input_messages || []).map((message, index) => (
                  <InputMessage key={`${message.role}-${index}`} message={message} index={index} />
                ))}
              </div>
            </section>

            <section>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-display font-bold uppercase tracking-wide text-nv-green">Output</h3>
                <span className="text-xs text-text-tertiary">
                  {countWords(outputReasoning) + countWords(outputContent)} words
                </span>
              </div>
              <div className="space-y-3">
                <TextBubble label="Reasoning" value={outputReasoning} tone="reasoning" />
                <TextBubble label="Content" value={outputContent} tone="content" />
                {(selectedRecord.code_blocks || []).map((code, index) => (
                  <TextBubble key={index} label={`Code block ${index + 1}`} value={code} />
                ))}
              </div>
            </section>
          </div>
        ) : null}
      </div>
    </div>
  );
}
