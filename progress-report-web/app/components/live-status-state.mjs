export function mergeStatusState(current, incoming) {
  if (incoming.state !== "unavailable") {
    return { snapshot: incoming, issue: null };
  }

  const lastValid =
    current.snapshot?.state !== "unavailable"
      ? current.snapshot
      : null;
  return {
    snapshot: lastValid ?? incoming,
    issue: incoming.message,
  };
}
