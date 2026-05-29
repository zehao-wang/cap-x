import { useEffect, useState } from 'react';
import type { ResetWizardEvent, ResetWizardAction } from '../types/messages';

interface ResetWizardModalProps {
  wizard: ResetWizardEvent | null;
  onAction: (action: ResetWizardAction) => void;
}

const STEP_TITLES: Record<string, string> = {
  connection: '步骤 1 / 3 · 检查机械臂连接',
  rest_pose: '步骤 2 / 3 · 回归 rest pose',
  rearrange: '步骤 3 / 3 · 重新摆放环境',
  complete: '真机 reset 完成',
};

const ACTION_LABELS: Record<ResetWizardAction, string> = {
  ready: 'Ready',
  confirm: '✓ 已就绪',
  reject: '✗ 未就绪',
};

// Button color per action: confirm=accent, ready=accent, reject=neutral/danger.
function actionClass(action: ResetWizardAction): string {
  if (action === 'reject') {
    return 'bg-surface-overlay border border-surface-border-light text-text-primary hover:bg-surface-raised';
  }
  return 'bg-accent text-surface-sunken hover:bg-accent-light';
}

export function ResetWizardModal({ wizard, onAction }: ResetWizardModalProps) {
  // Lock buttons right after a press until the next wizard event arrives, so a
  // double-click can't queue a second response that auto-advances a later step.
  const [submitted, setSubmitted] = useState(false);
  useEffect(() => {
    setSubmitted(false);
  }, [wizard]);

  if (!wizard) return null;

  const title = STEP_TITLES[wizard.step] ?? '真机 reset';
  const disabled = wizard.busy || submitted;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="w-[min(90vw,28rem)] rounded-lg border border-surface-border bg-surface-raised shadow-2xl animate-scale-in">
        <div className="border-b border-surface-border px-5 py-3">
          <h2 className="text-sm font-semibold text-accent">{title}</h2>
        </div>

        <div className="px-5 py-5">
          <p className="text-sm leading-6 text-text-primary whitespace-pre-line">
            {wizard.message}
          </p>

          {wizard.busy && (
            <div className="mt-4 flex items-center gap-2 text-xs text-text-secondary">
              <svg
                className="w-4 h-4 animate-spin text-accent"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z"
                />
              </svg>
              请稍候…
            </div>
          )}
        </div>

        {wizard.actions.length > 0 && (
          <div className="flex justify-end gap-2 border-t border-surface-border px-5 py-3">
            {wizard.actions.map((action) => (
              <button
                key={action}
                type="button"
                disabled={disabled}
                onClick={() => {
                  setSubmitted(true);
                  onAction(action);
                }}
                className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed ${actionClass(action)}`}
              >
                {ACTION_LABELS[action]}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
