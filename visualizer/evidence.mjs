// Recorded ONNX results are separate from the browser's diagnostic inference.
export async function readEvidence(fetcher, recipe, sha256) {
  const response=await fetcher('assets/evidence.json');
  if(response.ok){
    const bytes=await response.arrayBuffer();
    if(!recipe?.evidence_sha256||await sha256(bytes)!==recipe.evidence_sha256)throw new Error('Evidence identity mismatch');
    const result=JSON.parse(new TextDecoder().decode(bytes));
    if(!recipe.source_binding||result.binding!==recipe.source_binding||result.deployment_binding!==recipe.source_deployment_binding||!Array.isArray(result.rows))throw new Error('Evidence belongs to another model');
    return result.rows;
  }
  if(recipe?.source_deployment_binding)throw new Error('Measured results are not available for this model yet.');
  const historical=await fetcher('../final_suite/data/evidence.json.gz');
  if(!historical.ok)throw new Error('Restore generated evidence to load the exhaustive table.');
  const stream=historical.body.pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text()).rows;
}
