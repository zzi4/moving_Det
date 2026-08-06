"use client";

import { useState } from "react";

import type { PipelineLayer } from "../pipeline-story-data";
import { selectPipelineLayer } from "./pipeline-layer-state.mjs";

export function PipelineVisual({
  layers,
  caption,
  eager = false,
}: {
  layers: readonly PipelineLayer[];
  caption: string;
  eager?: boolean;
}) {
  const [selectedId, setSelectedId] = useState(layers[0].id);
  const [failedPath, setFailedPath] = useState<string | null>(null);
  const selected = selectPipelineLayer(layers, selectedId);

  function selectLayer(id: string) {
    setSelectedId(id);
    setFailedPath(null);
  }

  return (
    <figure className="pipeline-story-visual pipeline-visual">
      <div className="pipeline-layer-controls" aria-label={`${caption}图层`}>
        {layers.map((layer) => (
          <button
            type="button"
            aria-pressed={layer.id === selected.id}
            onClick={() => selectLayer(layer.id)}
            key={layer.id}
          >
            {layer.label}
          </button>
        ))}
      </div>

      {failedPath ? (
        <div className="pipeline-visual-fallback" role="status">
          <strong>证据图加载失败</strong>
          <p>{selected.alt}</p>
          <code>{failedPath}</code>
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={selected.id}
          src={selected.src}
          srcSet={`${selected.src1x} 640w, ${selected.src} 1280w`}
          sizes="(max-width: 900px) 100vw, 54vw"
          alt={selected.alt}
          width="1280"
          height="720"
          loading={eager ? "eager" : "lazy"}
          onError={() => setFailedPath(selected.src)}
        />
      )}

      <figcaption>{selected.caption}</figcaption>
    </figure>
  );
}
