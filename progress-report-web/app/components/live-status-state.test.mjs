import assert from "node:assert/strict";
import test from "node:test";

import { mergeStatusState } from "./live-status-state.mjs";

const valid = {
  state: "running",
  current_method: "multiscale",
  latest_frame: 65,
  message: "正在生成缓存",
};

test("keeps the last valid snapshot when collection becomes unavailable", () => {
  const current = { snapshot: valid, issue: null };
  const next = mergeStatusState(current, {
    state: "unavailable",
    current_method: null,
    latest_frame: null,
    message: "状态读取失败：EACCES",
  });

  assert.equal(next.snapshot, valid);
  assert.equal(next.issue, "状态读取失败：EACCES");
});

test("shows unavailable data when there is no previous valid snapshot", () => {
  const unavailable = {
    state: "unavailable",
    message: "状态读取失败",
  };
  const next = mergeStatusState(
    { snapshot: null, issue: null },
    unavailable,
  );

  assert.equal(next.snapshot, unavailable);
  assert.equal(next.issue, "状态读取失败");
});

test("clears an earlier issue after the next valid refresh", () => {
  const next = mergeStatusState(
    { snapshot: valid, issue: "网络错误" },
    { ...valid, latest_frame: 66 },
  );

  assert.equal(next.snapshot.latest_frame, 66);
  assert.equal(next.issue, null);
});
