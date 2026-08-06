export function selectPipelineLayer(layers, requestedId) {
  if (!Array.isArray(layers) || layers.length === 0) {
    throw new TypeError("layers must be a non-empty array");
  }
  return layers.find((layer) => layer.id === requestedId) ?? layers[0];
}
