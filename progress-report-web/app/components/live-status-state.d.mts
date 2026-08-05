export interface StatusLike {
  state: string;
  message: string;
}

export interface LiveStatusView<T extends StatusLike> {
  snapshot: T | null;
  issue: string | null;
}

export function mergeStatusState<T extends StatusLike>(
  current: LiveStatusView<T>,
  incoming: T,
): LiveStatusView<T>;
