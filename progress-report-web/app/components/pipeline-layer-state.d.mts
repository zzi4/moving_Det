export type PipelineLayerLike = {
  id: string;
};

export function selectPipelineLayer<T extends PipelineLayerLike>(
  layers: readonly T[],
  requestedId: string,
): T;
