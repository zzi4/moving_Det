import assert from "node:assert/strict";
import test from "node:test";

import { selectPipelineLayer } from "./pipeline-layer-state.mjs";

const layers = [
  {
    id: "before",
    label: "处理前",
    src: "/before.webp",
    src1x: "/before-1x.webp",
    alt: "处理前",
    caption: "处理前结果",
  },
  {
    id: "after",
    label: "处理后",
    src: "/after.webp",
    src1x: "/after-1x.webp",
    alt: "处理后",
    caption: "处理后结果",
  },
];

test("selects the requested pipeline evidence layer", () => {
  assert.equal(selectPipelineLayer(layers, "after").id, "after");
});

test("falls back to the first layer when the requested id is absent", () => {
  assert.equal(selectPipelineLayer(layers, "missing").id, "before");
});

test("rejects an empty layer collection", () => {
  assert.throws(
    () => selectPipelineLayer([], "before"),
    /layers must be a non-empty array/,
  );
});
